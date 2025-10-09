"""Sampling utilities for Qwen2.5-VL models.

# tiny helpers for quick sampling
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

# keeps warm starts fast
try:
    from jax.experimental.compilation_cache import compilation_cache as _cc  # type: ignore
    (getattr(_cc, "set_cache_dir", getattr(_cc, "initialize_cache", lambda *_: None)))(
        ".jax_cache"
    )
except Exception:
    pass

from vlmrl.models.qwen25vl import (
    KVCache,
    MRopeCtx,
    Qwen25VLModel,
    TextRopeCtx,
    build_mrope,
    build_text_rope,
    get_rope_index,
    create_model_from_ckpt,
)

from vlmrl.utils.vlm import (
    preprocess_image,
    decode_tokens,
    chat_prompt_with_image,
    extract_assistant,
    token_positions,
    mask_logits_topk_topp,
)

class Sampler:
    def __init__(self, model: Qwen25VLModel, params) -> None:
        # lightweight wrapper over the model
        self.model = model
        self.params = params
        text_spec = model.spec.text
        self._rope_section = tuple(text_spec.rope_section)
        self._rope_theta = float(text_spec.rope_theta)
        self._dtype = model.dtype
        # derived hyperparams for cache shape
        self._num_layers = text_spec.num_hidden_layers
        self._num_kv_heads = text_spec.num_key_value_heads
        self._head_dim = text_spec.head_dim

    def _init_cache(self, batch: int, max_len: int) -> KVCache:
        # pre-alloc kv cache
        return KVCache.init(
            batch=batch,
            num_layers=self._num_layers,
            num_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            max_len=max_len,
            dtype=self._dtype,
        )

    def prefill_text(
        self,
        prompt_tokens: np.ndarray | jnp.ndarray,
        pad_id: int,
        max_cache_len: Optional[int] = None,
    ) -> KVCache:
        tokens = jnp.asarray(prompt_tokens, dtype=jnp.int32)
        if tokens.ndim != 2:
            raise ValueError("prompt_tokens must have shape [batch, seq]")
        # standard text-only prefill
        positions, mask = token_positions(tokens, pad_id)
        cos, sin = build_text_rope(positions, self._rope_section, self._rope_theta, dtype=self._dtype)
        cache = self._init_cache(tokens.shape[0], max_cache_len or tokens.shape[1])

        @jax.jit
        def _prefill(params, tokens, cos, sin, mask, cache):
            _, cache_out = self.model.apply(
                {"params": params},
                tokens,
                cos,
                sin,
                mask=mask,
                cache=cache,
                method=self.model.forward_text,
            )
            return cache_out

        return _prefill(self.params, tokens, cos, sin, mask, cache)

    def prefill_vlm(
        self,
        prompt_tokens: np.ndarray | jnp.ndarray,
        vision_embeds: jnp.ndarray,
        image_pad_id: int,
        grid_thw: jnp.ndarray,
        pad_id: int,
        max_cache_len: Optional[int] = None,
    ) -> tuple[jnp.ndarray, KVCache, jnp.ndarray]:
        if self.model.spec.vision is None:
            raise ValueError("Model has no vision backbone configured")
        tokens = jnp.asarray(prompt_tokens, dtype=jnp.int32)
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("prefill_vlm currently supports batch=1")
        # mrope positions for interleaved vision tokens
        mask = (tokens != pad_id).astype(jnp.int32)
        pos3, deltas = get_rope_index(
            spatial_merge_size=self.model.spec.vision.spatial_merge_size,
            input_ids=tokens,
            image_grid_thw=grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=mask,
            tokens_per_second=float(self.model.spec.vision.tokens_per_second),
        )
        cos, sin = build_mrope(pos3, self._rope_section, self._rope_theta, dtype=self._dtype)
        max_len = max_cache_len or tokens.shape[1]  # quick cap
        cache = self._init_cache(tokens.shape[0], max_len)
        vision_embeds = jnp.asarray(vision_embeds, dtype=self._dtype)

        @jax.jit
        def _prefill_vlm(params, tokens, vision_embeds, image_pad_id, cos, sin, mask, cache):
            logits, cache_out = self.model.apply(
                {"params": params},
                tokens,
                vision_embeds,
                image_pad_id,
                cos,
                sin,
                mask=mask,
                cache=cache,
                method=self.model.forward_vlm,
            )
            return logits, cache_out

        logits, cache = _prefill_vlm(
            self.params, tokens, vision_embeds, image_pad_id, cos, sin, mask, cache
        )
        return logits, cache, deltas

    def decode_loop(
        self,
        cache: KVCache,
        first_token: jnp.ndarray,
        steps: int,
        *,
        temperature: float,
        top_p: Optional[float],
        eos_id: Optional[int],
        top_k: Optional[int] = None,
        rope_ctx: Union[TextRopeCtx, MRopeCtx],
        rng: jax.Array,
        return_logprobs: bool = False,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        if steps <= 0:
            # nothing to do
            empty = jnp.zeros((first_token.shape[0], 0), dtype=jnp.int32)
            if return_logprobs:
                return empty, empty.astype(jnp.float32)
            return empty, None

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        temp = jnp.float32(temperature)
        eos_scalar = jnp.int32(eos_id if eos_id is not None else -1)
        has_eos = eos_scalar >= 0
        use_top_k = int(top_k) if top_k is not None else 0
        topp_val_py = (
            float(top_p) if (top_p is not None and 0.0 < float(top_p) < 1.0) else None
        )

        def _one_step(params, rope_ctx, carry, _):
            # one decode step
            cache_state, current_tok, rng_state, stopped = carry
            rng_state, step_key = jax.random.split(rng_state)
            logits, cache_state = self.model.apply(
                {"params": params},
                current_tok,
                cache_state,
                rope_ctx,
                method=self.model.decode_step,
            )
            logits = logits.astype(jnp.float32) / temp  # apply temperature
            masked = mask_logits_topk_topp(logits, top_k=use_top_k, top_p=topp_val_py)  # shortlist
            next_token = jax.random.categorical(step_key, masked)
            if return_logprobs:
                log_probs = jax.nn.log_softmax(masked)
                gathered = log_probs[jnp.arange(log_probs.shape[0]), next_token]
            else:
                gathered = jnp.zeros((masked.shape[0],), dtype=jnp.float32)

            hit_eos = jnp.logical_and(has_eos, next_token == eos_scalar)  # cheap stop
            stopped_new = jnp.logical_or(stopped, hit_eos)
            effective_next = jnp.where(
                jnp.logical_and(stopped, has_eos),
                jnp.broadcast_to(eos_scalar, next_token.shape),
                next_token,
            )
            carry_out = (cache_state, effective_next.astype(jnp.int32), rng_state, stopped_new)
            y = (effective_next.astype(jnp.int32), gathered.astype(jnp.float32))
            return carry_out, y

        @jax.jit  # pack scan for speed
        def _scan_decode(params, rope_ctx, cache_init, first_tok_init, rng_init):
            init_carry = (
                cache_init,
                first_tok_init.astype(jnp.int32),
                rng_init,
                jnp.zeros_like(first_tok_init, dtype=jnp.bool_),
            )
            carry_out, ys = jax.lax.scan(
                lambda c, _x: _one_step(params, rope_ctx, c, _x), init_carry, xs=None, length=int(steps)
            )
            tokens_seq, logprobs_seq = ys  # each is (steps, batch)
            return tokens_seq.transpose(1, 0), logprobs_seq.transpose(1, 0)

        tokens_seq, logprobs_seq = _scan_decode(self.params, rope_ctx, cache, first_token, rng)
        if return_logprobs:
            return tokens_seq, logprobs_seq
        return tokens_seq, None

    def sample_text(
        self,
        prompt_tokens: np.ndarray | jnp.ndarray,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        rng: jax.Array,
        return_logprobs: bool = False,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        tokens = jnp.asarray(prompt_tokens, dtype=jnp.int32)
        if tokens.ndim != 2:
            raise ValueError("prompt_tokens must have shape [batch, seq]")
        # text path
        cache = self.prefill_text(tokens, pad_id, max_cache_len=tokens.shape[1] + max_new_tokens)
        lengths = cache.lengths.astype(jnp.int32)
        last_token_idx = jnp.maximum(lengths - 1, 0)
        last_token = jnp.take_along_axis(tokens, last_token_idx[:, None], axis=1).squeeze(1)
        rope_ctx = TextRopeCtx(self._rope_section, self._rope_theta, dtype=self._dtype)
        return self.decode_loop(
            cache,
            last_token,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_id=eos_id,
            rope_ctx=rope_ctx,
            rng=rng,
            return_logprobs=return_logprobs,
        )

    def sample_vlm(
        self,
        prompt_tokens: np.ndarray | jnp.ndarray,
        vision_embeds: jnp.ndarray,
        grid_thw: jnp.ndarray,
        image_pad_id: int,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        rng: jax.Array,
        return_logprobs: bool = False,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        logits, cache, deltas = self.prefill_vlm(
            prompt_tokens,
            vision_embeds,
            image_pad_id,
            grid_thw,
            pad_id,
            max_cache_len=prompt_tokens.shape[1] + max_new_tokens,
        )
        # seed first token from last prefill logits
        last_logits = logits[:, -1, :].astype(jnp.float32)
        temp = jnp.float32(temperature)
        last_logits = last_logits / temp
        last_logits = mask_logits_topk_topp(last_logits, top_k=top_k, top_p=top_p)
        rng, first_key = jax.random.split(rng)
        first_token = jax.random.categorical(first_key, last_logits)
        if return_logprobs:
            first_logprob = jax.nn.log_softmax(last_logits)[jnp.arange(last_logits.shape[0]), first_token]
        else:
            first_logprob = None

        # dynamic mRoPE context using computed deltas
        rope_ctx = MRopeCtx(self._rope_section, self._rope_theta, deltas.astype(jnp.int32), dtype=self._dtype)
        steps = max_new_tokens - 1
        decode_tokens, decode_logprobs = self.decode_loop(
            cache,
            first_token.astype(jnp.int32),
            steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_id=eos_id,
            rope_ctx=rope_ctx,
            rng=rng,
            return_logprobs=return_logprobs,
        )

        generated = (
            jnp.concatenate([first_token[:, None], decode_tokens], axis=1)
            if decode_tokens.size > 0
            else first_token[:, None]
        )
        if return_logprobs:
            if first_logprob is None:
                raise RuntimeError("first_logprob must be available when return_logprobs=True")
            combined = (
                jnp.concatenate([first_logprob[:, None], decode_logprobs], axis=1)
                if decode_logprobs is not None and decode_logprobs.size > 0
                else first_logprob[:, None]
            )
            return generated, combined
        return generated, None


def _resolve_image_pad_id(tokenizer, ckpt_dir: str) -> int:
    try:
        image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")  # try tokenizer first
        if isinstance(image_pad, (list, tuple)):
            image_pad = image_pad[0]
        if image_pad is not None and int(image_pad) >= 0:
            return int(image_pad)
    except Exception:
        pass

    try:
        with open(f"{ckpt_dir}/config.json", "r") as f:  # fallback to config
            cfg = json.load(f)
        return int(cfg.get("image_token_id", 151655))
    except Exception:
        return 151655


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Lightweight sampler for Qwen2.5-VL checkpoints.")
    parser.add_argument("--ckpt_dir", "--ckpt-dir", dest="ckpt_dir", type=str, required=True, help="Checkpoint directory produced by hf_to_jax.")
    parser.add_argument("--prompt", type=str, default=None, help="User prompt. Required for text or vision runs.")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for VLM sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None, help="Top-k shortlist for faster sampling (optional)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad_id", type=int, default=None, help="Override tokenizer pad id.")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.ckpt_dir}...")  # basic cli
    model, params = create_model_from_ckpt(args.ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=False)

    pad_id = (
        args.pad_id
        if args.pad_id is not None
        else (getattr(tokenizer, "pad_token_id", None) or model.spec.pad_token_id or 0)
    )
    eos_id = getattr(tokenizer, "eos_token_id", None) or model.spec.eos_token_id
    if eos_id is None:
        raise SystemExit("EOS token id not found; please provide a tokenizer with eos_token_id.")

    sampler = Sampler(model, params)
    rng = jax.random.PRNGKey(args.seed)

    if args.image is not None:
        if model.spec.vision is None:
            raise SystemExit("This checkpoint has no vision backbone configured.")
        if not args.prompt:
            raise SystemExit("Provide --prompt when sampling with an image.")

        vision_spec = model.spec.vision
        pixel_values, grid_thw = preprocess_image(
            args.image,
            patch_size=vision_spec.patch_size,
            spatial_merge_size=vision_spec.spatial_merge_size,
            temporal_patch_size=vision_spec.temporal_patch_size,
        )

        vision_embeds = model.apply(
            {"params": params},
            pixel_values,
            grid_thw,
            method=model.encode_vision,
        )
        num_vision_tokens = int(vision_embeds.shape[0])
        prompt_text = chat_prompt_with_image(num_vision_tokens, args.prompt)
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

        image_pad_id = _resolve_image_pad_id(tokenizer, args.ckpt_dir)
        generated, _ = sampler.sample_vlm(
            prompt_tokens,
            vision_embeds,
            grid_thw,
            image_pad_id=image_pad_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_id=eos_id,
            pad_id=pad_id,
            rng=rng,
            return_logprobs=False,
        )
        new_tokens = generated[0].tolist()
        full_ids = input_ids + new_tokens
        full_text = tokenizer.decode(
            full_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        answer = extract_assistant(full_text) or decode_tokens(tokenizer, new_tokens)
        print(answer.strip())
        return

    # Text-only sampling
    if not args.prompt:
        raise SystemExit("Provide --prompt for text-only sampling.")

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    generated, _ = sampler.sample_text(
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_id=eos_id,
        pad_id=pad_id,
        rng=rng,
        return_logprobs=False,
    )
    new_tokens = generated[0].tolist()
    full_ids = input_ids + new_tokens
    full_text = tokenizer.decode(
        full_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    answer = extract_assistant(full_text) or decode_tokens(tokenizer, new_tokens)
    print(answer.strip())


__all__ = ["Sampler"]


if __name__ == "__main__":
    _run_cli()
