"""Unified sampling utilities for Qwen3-VL returning typed results.

This module targets the Qwen3-VL variant only. It provides a small, focused
sampler that accepts a SamplingConfig and returns a SampleResult, with a
single entry point that handles both text-only and vision-language (VLM)
sampling via the VLMInputs type.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np

from vlmrl.core.types import SamplingConfig, VLMInputs, SampleResult
from vlmrl.models.qwen3vl.model import KVCache, VisionEmbeddings, build_mrope, build_text_rope, get_rope_index
from vlmrl.utils.vlm import token_positions, mask_logits_topk_topp


@dataclass
class _RopeSpec:
    rope_section: tuple
    rope_theta: float
    rope_scaling_type: Optional[str]
    rope_scaling_factor: Optional[float]
    dtype: jnp.dtype
    num_layers: int
    num_kv_heads: int
    head_dim: int


def _rope_spec_from_model(model) -> _RopeSpec:
    text_spec = model.spec.text
    rope_section = tuple(text_spec.rope_section)
    dtype = model.dtype
    return _RopeSpec(
        rope_section=rope_section,
        rope_theta=float(text_spec.rope_theta),
        rope_scaling_type=getattr(text_spec, "rope_scaling_type", None),
        rope_scaling_factor=getattr(text_spec, "rope_scaling_factor", None),
        dtype=dtype,
        num_layers=int(text_spec.num_hidden_layers),
        num_kv_heads=int(text_spec.num_key_value_heads),
        head_dim=int(text_spec.head_dim),
    )


def _init_cache(spec: _RopeSpec, batch: int, max_len: int) -> KVCache:
    return KVCache.init(
        batch=batch,
        num_layers=spec.num_layers,
        num_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
        max_len=max_len,
        dtype=spec.dtype,
    )


def _prefill_text(model, params, tokens: jnp.ndarray, pad_id: int, spec: _RopeSpec, max_cache_len: Optional[int]) -> Tuple[KVCache, jnp.ndarray]:
    if tokens.ndim != 2:
        raise ValueError("prompt_tokens must have shape [batch, seq]")
    positions, mask = token_positions(tokens, pad_id)
    cos, sin = build_text_rope(
        positions,
        spec.rope_section,
        spec.rope_theta,
        dtype=spec.dtype,
        rope_scaling_type=spec.rope_scaling_type,
        rope_scaling_factor=spec.rope_scaling_factor,
    )
    cache = _init_cache(spec, tokens.shape[0], int(max_cache_len or tokens.shape[1]))

    @jax.jit
    def _prefill(params, tokens, cos, sin, mask, cache):
        out = model.apply({"params": params}, tokens, cos, sin, mask=mask, cache=cache, method=model.forward_text)
        # out: (logits, cache, attn?)
        return out[1]

    cache_out = _prefill(params, tokens, cos, sin, mask, cache)
    rope_deltas = jnp.zeros((tokens.shape[0], 1), dtype=jnp.int32)
    return cache_out, rope_deltas


def _prefill_vlm(
    model,
    params,
    tokens: jnp.ndarray,
    vision: Union[VisionEmbeddings, jnp.ndarray],
    grid_thw: jnp.ndarray,
    image_pad_id: int,
    pad_id: int,
    spec: _RopeSpec,
    max_cache_len: Optional[int],
) -> Tuple[jnp.ndarray, KVCache, jnp.ndarray]:
    if tokens.ndim != 2:
        raise ValueError("prefill_vlm expects tokens with shape [batch, seq]")
    mask = (tokens != pad_id).astype(jnp.int32)
    batch = int(tokens.shape[0])
    # Normalize grid to [B, N, 3]
    grid_thw = jnp.asarray(grid_thw, dtype=jnp.int32)
    if grid_thw.ndim == 2:
        if grid_thw.shape[0] == 1 and batch > 1:
            grid_thw = jnp.tile(grid_thw[None, ...], (batch, 1, 1))
        elif grid_thw.shape[0] == batch and grid_thw.shape[1] == 3:
            grid_thw = grid_thw[:, None, :]
    elif grid_thw.ndim == 3 and grid_thw.shape[0] == 1 and batch > 1:
        grid_thw = jnp.tile(grid_thw, (batch, 1, 1))

    # RoPE indices for Qwen3-VL
    pos3, deltas = get_rope_index(
        spatial_merge_size=model.spec.vision.spatial_merge_size,
        input_ids=tokens,
        image_grid_thw=grid_thw,
        attention_mask=mask,
    )

    cos, sin = build_mrope(
        pos3,
        spec.rope_section,
        spec.rope_theta,
        dtype=spec.dtype,
        rope_scaling_type=spec.rope_scaling_type,
        rope_scaling_factor=spec.rope_scaling_factor,
    )

    max_len = int(max_cache_len or tokens.shape[1])
    cache = _init_cache(spec, tokens.shape[0], max_len)

    if isinstance(vision, VisionEmbeddings):
        vision_pack = vision.cast(spec.dtype)
    else:
        vision_arr = jnp.asarray(vision, dtype=spec.dtype)
        vision_pack = VisionEmbeddings(tokens=vision_arr, deepstack=())

    @jax.jit
    def _prefill(params, tokens, vision_pack, image_pad_id, cos, sin, mask, cache):
        out = model.apply(
            {"params": params},
            tokens,
            vision_pack,
            image_pad_id,
            cos,
            sin,
            mask=mask,
            cache=cache,
            method=model.forward_vlm,
        )
        return out[0], out[1]

    logits, cache = _prefill(params, tokens, vision_pack, image_pad_id, cos, sin, mask, cache)
    rope_deltas = deltas.astype(jnp.int32)
    return logits, cache, rope_deltas


def _decode_loop(
    model,
    params,
    cache: KVCache,
    first_token: jnp.ndarray,
    steps: int,
    *,
    temperature: float,
    top_p: Optional[float],
    eos_id: Optional[int],
    top_k: Optional[int],
    rope_deltas: Optional[jnp.ndarray],
    rng: jax.Array,
    return_logprobs: bool,
    unroll: int = 1,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    if steps <= 0:
        empty = jnp.zeros((first_token.shape[0], 0), dtype=jnp.int32)
        return (empty, empty.astype(jnp.float32)) if return_logprobs else (empty, None)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    temp = jnp.float32(temperature)
    eos_scalar = jnp.int32(eos_id if eos_id is not None else -1)
    has_eos = eos_scalar >= 0
    use_top_k = int(top_k) if top_k is not None else 0
    topp_val_py = float(top_p) if (top_p is not None and 0.0 < float(top_p) < 1.0) else None

    # On METAL, buffer donation is not implemented; avoid donating cache arg.
    _is_metal = (jax.devices()[0].platform.lower() == "metal")
    _donate = () if _is_metal else (2,)

    @partial(jax.jit, donate_argnums=_donate)
    def _scan_decode(params, offsets, cache_init, first_tok_init, rng_init):
        def _one_step(params, offsets, carry, _):
            cache_state, current_tok, rng_state, stopped = carry
            rng_state, step_key = jax.random.split(rng_state)
            step_mask = (jnp.logical_not(stopped)).astype(jnp.int32)[:, None]
            logits, cache_state, _ = model.apply(
                {"params": params},
                current_tok,
                cache_state,
                offsets,
                step_mask,
                method=model.decode_step,
            )
            logits = logits.astype(jnp.float32) / temp
            # Guard against NaN/Inf on experimental backends: drop invalids
            logits = jnp.nan_to_num(
                logits,
                nan=jnp.float32(-1e9),
                neginf=jnp.float32(-1e9),
                posinf=jnp.float32(-1e9),
            )
            # On METAL, prefer top-k shortlist for top-p to avoid argsort issues
            eff_top_k = use_top_k
            if _is_metal and (use_top_k == 0) and (topp_val_py is not None):
                # Prefer a generous shortlist to approximate top-p stably on METAL
                eff_top_k = 1024
            masked = mask_logits_topk_topp(logits, top_k=eff_top_k, top_p=topp_val_py)
            # Fallback if everything got masked (degenerate shortlist)
            row_max = jnp.max(masked, axis=-1)
            need_fallback = (row_max <= jnp.float32(-1e8))[:, None]
            masked = jnp.where(need_fallback, logits, masked)
            next_token = jax.random.categorical(step_key, masked)
            if return_logprobs:
                log_probs = jax.nn.log_softmax(masked)
                gathered = log_probs[jnp.arange(log_probs.shape[0]), next_token]
            else:
                gathered = jnp.zeros((masked.shape[0],), dtype=jnp.float32)
            hit_eos = jnp.logical_and(has_eos, next_token == eos_scalar)
            stopped_new = jnp.logical_or(stopped, hit_eos)
            effective_next = jnp.where(jnp.logical_and(stopped, has_eos), jnp.broadcast_to(eos_scalar, next_token.shape), next_token)
            carry_out = (cache_state, effective_next.astype(jnp.int32), rng_state, stopped_new)
            y = (effective_next.astype(jnp.int32), gathered.astype(jnp.float32))
            return carry_out, y

        init_carry = (
            cache_init,
            first_tok_init.astype(jnp.int32),
            rng_init,
            jnp.zeros_like(first_tok_init, dtype=jnp.bool_),
        )
        carry_out, ys = jax.lax.scan(
            lambda c, _: _one_step(params, offsets, c, _),
            init_carry,
            xs=None,
            length=int(steps),
            unroll=int(max(1, unroll)),
        )
        tokens_seq, logprobs_seq = ys
        return tokens_seq.transpose(1, 0), logprobs_seq.transpose(1, 0)

    offsets = jnp.asarray(rope_deltas if rope_deltas is not None else jnp.zeros((cache.lengths.shape[0], 1), dtype=jnp.int32))
    return _scan_decode(params, offsets, cache, first_token, rng)


def _decode_loop_stepwise(
    model,
    params,
    cache: KVCache,
    first_token: jnp.ndarray,
    steps: int,
    *,
    temperature: float,
    top_p: Optional[float],
    eos_id: Optional[int],
    top_k: Optional[int],
    rope_deltas: Optional[jnp.ndarray],
    rng: jax.Array,
    return_logprobs: bool,
    step_callback: Optional[object] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Step-by-step decode where each step is a small jitted call.

    This avoids very long fused kernels that can time out on Metal.
    """
    if steps <= 0:
        empty = jnp.zeros((first_token.shape[0], 0), dtype=jnp.int32)
        return (empty, empty.astype(jnp.float32)) if return_logprobs else (empty, None)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    temp = jnp.float32(temperature)
    eos_scalar = jnp.int32(eos_id if eos_id is not None else -1)
    has_eos = eos_scalar >= 0
    use_top_k = int(top_k) if top_k is not None else 0
    topp_val_py = float(top_p) if (top_p is not None and 0.0 < float(top_p) < 1.0) else None

    offsets = jnp.asarray(rope_deltas if rope_deltas is not None else jnp.zeros((cache.lengths.shape[0], 1), dtype=jnp.int32))

    # On METAL, buffer donation is not implemented; avoid donating cache arg.
    _is_metal = (jax.devices()[0].platform.lower() == "metal")
    _donate = () if _is_metal else (2,)

    # Mark `enable_attn` as static so Python control flow depending on it
    # (e.g., inside model.apply -> attention) is resolved at trace time.
    @partial(jax.jit, donate_argnums=_donate, static_argnums=(6,))
    def _one_step(params, offsets, cache_state, current_tok, rng_state, stopped, enable_attn: bool):
        rng_state, step_key = jax.random.split(rng_state)
        step_mask = (jnp.logical_not(stopped)).astype(jnp.int32)[:, None]
        logits, cache_state, attn_row = model.apply(
            {"params": params},
            current_tok,
            cache_state,
            offsets,
            step_mask,
            collect_attn=enable_attn,
            method=model.decode_step,
        )
        logits = logits.astype(jnp.float32) / temp
        # Guard against NaN/Inf on experimental backends: drop invalids
        logits = jnp.nan_to_num(
            logits,
            nan=jnp.float32(-1e9),
            neginf=jnp.float32(-1e9),
            posinf=jnp.float32(-1e9),
        )
        # On METAL, prefer top-k shortlist for top-p to avoid argsort issues
        eff_top_k = use_top_k
        if _is_metal and (use_top_k == 0) and (topp_val_py is not None):
            eff_top_k = 1024
        masked = mask_logits_topk_topp(logits, top_k=eff_top_k, top_p=topp_val_py)
        # Fallback if everything got masked (degenerate shortlist)
        row_max = jnp.max(masked, axis=-1)
        need_fallback = (row_max <= jnp.float32(-1e8))[:, None]
        masked = jnp.where(need_fallback, logits, masked)
        next_token = jax.random.categorical(step_key, masked)
        logprob_row = jax.nn.log_softmax(masked)
        gathered = logprob_row[jnp.arange(logprob_row.shape[0]), next_token]
        hit_eos = jnp.logical_and(has_eos, next_token == eos_scalar)
        stopped_new = jnp.logical_or(stopped, hit_eos)
        effective_next = jnp.where(jnp.logical_and(stopped, has_eos), jnp.broadcast_to(eos_scalar, next_token.shape), next_token)
        return cache_state, effective_next.astype(jnp.int32), rng_state, stopped_new, gathered.astype(jnp.float32), attn_row

    tokens_acc: List[jnp.ndarray] = []
    logprobs_acc: List[jnp.ndarray] = []
    tok = first_token.astype(jnp.int32)
    rng_state = rng
    stopped = jnp.zeros_like(tok, dtype=jnp.bool_)
    cache_state = cache
    # Attention summary config
    enable_attn = bool(step_callback is not None)
    for i in range(int(steps)):
        cache_state, tok, rng_state, stopped, logp, attn_row = _one_step(
            params, offsets, cache_state, tok, rng_state, stopped, enable_attn
        )
        tokens_acc.append(tok)
        if return_logprobs:
            logprobs_acc.append(logp)

        # Optional per-step callback (Python side) for live UIs
        if step_callback is not None:
            try:
                # Convert small scalars to Python types
                token_id = int(np.asarray(tok[0]).item())
                cache_len = int(np.asarray(cache_state.lengths[0]).item())
                cache_max = int(np.asarray(cache_state.keys.shape[3]).item())
                token_lp = float(np.asarray(logp[0]).item())
                rope_delta = int(np.asarray(offsets[0, 0]).item()) if offsets.ndim == 2 else int(np.asarray(offsets[0]).item())
                info = {
                    "step": int(i + 1),
                    "token_id": token_id,
                    "logprob": token_lp,
                    "cache_len": cache_len,
                    "cache_max": cache_max,
                    "rope_delta": rope_delta,
                }
                if enable_attn:
                    # attn_row: [N, H]
                    at = np.asarray(attn_row)
                    if at.size > 0:
                        info["attn_heads"] = at  # numpy array ok; callback converts/uses
                try:
                    step_callback(info)
                except Exception:
                    pass
            except Exception:
                pass

    new_tokens = jnp.stack(tokens_acc, axis=1)
    if return_logprobs:
        new_logprobs = jnp.stack(logprobs_acc, axis=1)
    else:
        new_logprobs = None
    return new_tokens, new_logprobs


def sample(
    model,
    params,
    inputs: Union[VLMInputs, jnp.ndarray, np.ndarray],
    cfg: SamplingConfig,
    rng: jax.Array,
    *,
    tokenizer=None,
    return_logprobs: bool = False,
    decode_impl: str = "scan",
    decode_unroll: int = 1,
    step_callback: Optional[object] = None,
) -> SampleResult:
    """Unified sampler entry point.

    If `inputs` is a VLMInputs instance, runs vision-language sampling. If it
    is a [B, T] token array, runs text-only sampling. Returns `SampleResult`.
    """
    spec = _rope_spec_from_model(model)

    # Normalize tokens to jnp
    if isinstance(inputs, VLMInputs):
        tokens = jnp.asarray(inputs.prompt_tokens, dtype=jnp.int32)
        # Prefill VLM
        _, cache, rope_deltas = _prefill_vlm(
            model,
            params,
            tokens,
            inputs.vision,
            inputs.grid_thw,
            inputs.image_pad_id,
            cfg.pad_id,
            spec,
            max_cache_len=int(tokens.shape[1] + cfg.max_new_tokens),
        )
    else:
        tokens = jnp.asarray(inputs, dtype=jnp.int32)
        # Prefill text
        cache, rope_deltas = _prefill_text(
            model,
            params,
            tokens,
            cfg.pad_id,
            spec,
            max_cache_len=int(tokens.shape[1] + cfg.max_new_tokens),
        )

    lengths = cache.lengths.astype(jnp.int32)
    last_token_idx = jnp.maximum(lengths - 1, 0)
    last_token = jnp.take_along_axis(tokens, last_token_idx[:, None], axis=1).squeeze(1)

    if decode_impl == "step":
        new_tokens, new_logprobs = _decode_loop_stepwise(
            model,
            params,
            cache,
            last_token,
            int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_p=cfg.top_p,
            eos_id=cfg.eos_id,
            top_k=cfg.top_k,
            rope_deltas=rope_deltas,
            rng=rng,
            return_logprobs=return_logprobs,
            step_callback=step_callback,
        )
    else:
        new_tokens, new_logprobs = _decode_loop(
            model,
            params,
            cache,
            last_token,
            int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_p=cfg.top_p,
            eos_id=cfg.eos_id,
            top_k=cfg.top_k,
            rope_deltas=rope_deltas,
            rng=rng,
            return_logprobs=return_logprobs,
            unroll=int(max(1, decode_unroll)),
        )

    texts: List[str] = []
    if tokenizer is not None:
        try:
            # Decode generated tokens only
            for row in np.asarray(new_tokens):
                texts.append(tokenizer.decode(row.tolist(), skip_special_tokens=True))
        except Exception:
            texts = [""] * int(new_tokens.shape[0])

    return SampleResult(tokens=new_tokens, logprobs=new_logprobs, texts=texts)


__all__ = ["sample"]
