"""Environment-aware sampling helpers shared by PPO and evaluation utilities.

This module centralizes how we:
  1. Convert environment observations into VLM-friendly prompts and embeddings.
  2. Invoke the unified sampler on those prompts.
  3. Package the resulting generations alongside rewards and metadata.

By keeping this logic here, we avoid duplicating it across PPO, smoke tests,
or one-off evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Apply macOS Metal-friendly env vars before importing jax.
try:
    from vlmrl.utils.platform import apply_macos_env, is_macos

    apply_macos_env()
except Exception:
    def is_macos() -> bool:  # type: ignore
        return False

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from vlmrl.core.batching import (
    build_masks,
    pad_float_sequences,
    pad_grid,
    pad_sequences,
    pad_vision,
)
from vlmrl.core.policy import build_rope
from vlmrl.core.sampling import sample as unified_sample
from vlmrl.core.types import Batch, Rollout, SamplingConfig, VLMInputs
from vlmrl.envs.base import BaseEnv, BaseState, create_env
from vlmrl.models.qwen3vl.model import Qwen3VLModel, VisionEmbeddings, create_model_from_ckpt
from vlmrl.utils.vlm import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    chat_prompt_with_image,
    chat_prompt_with_images,
    preprocess_image,
)


@dataclass(frozen=True)
class PreparedPrompt:
    """Vision-aware prompt ready to feed the sampler."""

    prompt_tokens: jnp.ndarray  # Shape [1, seq]
    prompt_length: int
    vision: Union[VisionEmbeddings, jnp.ndarray]
    grid: jnp.ndarray
    prompt_text: str


@dataclass(frozen=True)
class SampledAction:
    """Sampler outputs aligned with the prompt that produced them."""

    prompt_tokens: jnp.ndarray  # Shape [seq]
    prompt_length: int
    full_tokens: jnp.ndarray  # Prompt + action tokens
    full_logprobs: jnp.ndarray  # Zeros for prompt prefix
    action_tokens: List[int]
    action_logprobs: jnp.ndarray
    action_length: int
    text: str
    vision: Union[VisionEmbeddings, jnp.ndarray]
    grid: jnp.ndarray


@dataclass
class Episode:
    """Single environment rollout paired with sampler metadata."""

    state: BaseState
    observation: Any
    prompt_text: str
    prompt_tokens: jnp.ndarray
    prompt_length: int
    full_tokens: jnp.ndarray
    full_logprobs: jnp.ndarray
    action_tokens: List[int]
    action_logprobs: jnp.ndarray
    action_length: int
    generated_text: str
    vision: Union[VisionEmbeddings, jnp.ndarray]
    grid: jnp.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class EpisodeBatch:
    """Batched rollout data returned by :func:`collect_episodes`."""

    episodes: List[Episode]
    raw_info: Dict[str, List[Any]]

    def rewards_array(self) -> np.ndarray:
        return np.asarray([ep.reward for ep in self.episodes], dtype=np.float32)


def _resolve_pixels(val: Optional[int], fallback: int) -> int:
    """Convert optional pixel bounds into positive integers with sensible fallbacks."""
    if val is None:
        return fallback
    try:
        as_int = int(val)
    except (TypeError, ValueError):
        return fallback
    return as_int if as_int > 0 else fallback


def _prepare_prompt(
    model: Qwen3VLModel,
    params: Any,
    tokenizer,
    obs: Any,
    *,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> PreparedPrompt:
    """Encode environment observation into prompt tokens + vision embeddings."""
    vision_spec = model.spec.vision
    if vision_spec is None:
        raise ValueError("Model must include a vision backbone for vision environments.")

    min_px = _resolve_pixels(min_pixels, DEFAULT_MIN_PIXELS)
    max_px = _resolve_pixels(max_pixels, DEFAULT_MAX_PIXELS)

    patch_size = vision_spec.patch_size
    spatial_merge = vision_spec.spatial_merge_size
    temporal_patch = vision_spec.temporal_patch_size

    if hasattr(obs, "image_left") and hasattr(obs, "image_right") and hasattr(obs, "statement"):
        # Multi-image observation (e.g., NLVR2).
        pix_l, grid_l = preprocess_image(
            getattr(obs, "image_left"),
            patch_size=patch_size,
            spatial_merge_size=spatial_merge,
            temporal_patch_size=temporal_patch,
            min_pixels=min_px,
            max_pixels=max_px,
        )
        pix_r, grid_r = preprocess_image(
            getattr(obs, "image_right"),
            patch_size=patch_size,
            spatial_merge_size=spatial_merge,
            temporal_patch_size=temporal_patch,
            min_pixels=min_px,
            max_pixels=max_px,
        )
        emb_l = model.apply({"params": params}, pix_l, grid_l, method=model.encode_vision)
        emb_r = model.apply({"params": params}, pix_r, grid_r, method=model.encode_vision)
        if isinstance(emb_l, VisionEmbeddings) and isinstance(emb_r, VisionEmbeddings):
            vision_pack = VisionEmbeddings.concatenate([emb_l, emb_r])
            counts = [int(emb_l.tokens.shape[0]), int(emb_r.tokens.shape[0])]
        else:
            vision_pack = jnp.concatenate([emb_l, emb_r], axis=0)
            counts = [int(emb_l.shape[0]), int(emb_r.shape[0])]
        grid = jnp.concatenate([grid_l, grid_r], axis=0)
        question = f"Look at the two images. {getattr(obs, 'statement', '')}"
        prompt_text = chat_prompt_with_images(counts, question)
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)
        return PreparedPrompt(
            prompt_tokens=prompt_tokens,
            prompt_length=int(prompt_tokens.shape[1]),
            vision=vision_pack,
            grid=grid,
            prompt_text=prompt_text,
        )

    # Default single-image observation path.
    image_src = getattr(obs, "image", None) or getattr(obs, "image_path", None)
    if image_src is None:
        raise ValueError("Observation must expose `image` or `image_path` for vision sampling.")

    pix, grid = preprocess_image(
        image_src,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge,
        temporal_patch_size=temporal_patch,
        min_pixels=min_px,
        max_pixels=max_px,
    )
    vision_pack = model.apply({"params": params}, pix, grid, method=model.encode_vision)
    if isinstance(vision_pack, VisionEmbeddings):
        num_tokens = int(vision_pack.tokens.shape[0])
    else:
        num_tokens = int(vision_pack.shape[0])
    question = getattr(obs, "question", "")
    prompt_text = chat_prompt_with_image(num_tokens, question)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)
    return PreparedPrompt(
        prompt_tokens=prompt_tokens,
        prompt_length=int(prompt_tokens.shape[1]),
        vision=vision_pack,
        grid=grid,
        prompt_text=prompt_text,
    )


def _sample_prompt(
    model: Qwen3VLModel,
    params: Any,
    tokenizer,
    prepared: PreparedPrompt,
    sampling_cfg: SamplingConfig,
    *,
    image_pad_id: int,
    rng: jax.Array,
    return_logprobs: bool,
    max_sequence_length: Optional[int],
    decode_impl: str = "scan",
    decode_unroll: int = 1,
    step_callback: Optional[object] = None,
) -> SampledAction:
    """Run the unified sampler on a prepared prompt and normalize outputs."""
    inputs = VLMInputs(
        prompt_tokens=prepared.prompt_tokens,
        vision=prepared.vision,
        grid_thw=prepared.grid,
        image_pad_id=image_pad_id,
    )
    result = unified_sample(
        model,
        params,
        inputs,
        sampling_cfg,
        rng=rng,
        tokenizer=tokenizer,
        return_logprobs=return_logprobs,
        decode_impl=decode_impl,
        decode_unroll=int(max(1, decode_unroll)),
        step_callback=step_callback,
    )

    gen_tokens = result.tokens[0]
    if return_logprobs:
        if result.logprobs is None:
            gen_logprobs = jnp.zeros_like(gen_tokens, dtype=jnp.float32)
        else:
            gen_logprobs = result.logprobs[0]
    else:
        gen_logprobs = (
            result.logprobs[0]
            if result.logprobs is not None
            else jnp.zeros((gen_tokens.shape[0],), dtype=jnp.float32)
        )

    prompt_tokens_flat = prepared.prompt_tokens[0]
    prompt_len = int(prompt_tokens_flat.shape[0])
    prompt_tokens_eff = prompt_tokens_flat
    prompt_len_eff = prompt_len
    gen_tokens_eff = gen_tokens
    gen_logprobs_eff = gen_logprobs

    total_len = int(prompt_tokens_flat.shape[0] + gen_tokens.shape[0])
    if max_sequence_length is not None and total_len > int(max_sequence_length):
        max_seq = int(max_sequence_length)
        prompt_slice = max(0, min(prompt_len, max_seq))
        prompt_tokens_eff = prompt_tokens_flat[:prompt_slice]
        prompt_len_eff = int(prompt_tokens_eff.shape[0])
        allowed = max_seq - prompt_len
        if allowed <= 0:
            gen_tokens_eff = gen_tokens[:0]
            gen_logprobs_eff = gen_logprobs[:0]
        else:
            gen_tokens_eff = gen_tokens[:allowed]
            gen_logprobs_eff = gen_logprobs[:allowed]

    full_tokens = jnp.concatenate([prompt_tokens_eff, gen_tokens_eff], axis=0)
    zeros = jnp.zeros((prompt_len_eff,), dtype=jnp.float32)
    full_logprobs = jnp.concatenate([zeros, gen_logprobs_eff], axis=0)

    action_tokens = np.asarray(gen_tokens_eff, dtype=np.int32).tolist()
    action_length = int(gen_tokens_eff.shape[0])
    text = result.texts[0] if result.texts else ""

    return SampledAction(
        prompt_tokens=prompt_tokens_eff,
        prompt_length=prompt_len_eff,
        full_tokens=full_tokens,
        full_logprobs=full_logprobs,
        action_tokens=action_tokens,
        action_logprobs=gen_logprobs_eff,
        action_length=action_length,
        text=text,
        vision=prepared.vision,
        grid=prepared.grid,
    )


def _transpose_info(info: Dict[str, List[Any]], batch_size: int) -> List[Dict[str, Any]]:
    """Convert env.step_list info dict into per-episode dictionaries."""
    per_episode = [dict() for _ in range(batch_size)]
    for key, values in info.items():
        for idx in range(batch_size):
            if idx < len(values):
                per_episode[idx][key] = values[idx]
    return per_episode


def collect_episodes(
    env: BaseEnv,
    tokenizer,
    model: Qwen3VLModel,
    params: Any,
    sampling_cfg: SamplingConfig,
    *,
    image_pad_id: int,
    batch_size: int,
    rng: jax.Array,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
    return_logprobs: bool = True,
    decode_impl: str = "scan",
    decode_unroll: int = 1,
    step_callback: Optional[object] = None,
) -> EpisodeBatch:
    """Collect a batch of environment episodes using the unified sampler."""
    states: List[BaseState] = []
    observations: List[Any] = []
    prepared_prompts: List[PreparedPrompt] = []
    for offset in range(batch_size):
        state, obs = env.reset(offset)
        states.append(state)
        observations.append(obs)
        prepared_prompts.append(
            _prepare_prompt(
                model,
                params,
                tokenizer,
                obs,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        )

    keys = jax.random.split(rng, batch_size)
    sampled: List[SampledAction] = []
    for prep, key in zip(prepared_prompts, keys, strict=True):
        sampled.append(
            _sample_prompt(
                model,
                params,
                tokenizer,
                prep,
                sampling_cfg,
                image_pad_id=image_pad_id,
                rng=key,
                return_logprobs=return_logprobs,
                max_sequence_length=max_sequence_length,
                decode_impl=decode_impl,
                decode_unroll=decode_unroll,
                step_callback=step_callback,
            )
        )

    action_lists = [act.action_tokens for act in sampled]
    _, _, rewards, dones, info = env.step_list(states, action_lists)
    info_per_episode = _transpose_info(info, batch_size)

    episodes: List[Episode] = []
    for idx in range(batch_size):
        episodes.append(
            Episode(
                state=states[idx],
                observation=observations[idx],
                prompt_text=prepared_prompts[idx].prompt_text,
                prompt_tokens=sampled[idx].prompt_tokens,
                prompt_length=sampled[idx].prompt_length,
                full_tokens=sampled[idx].full_tokens,
                full_logprobs=sampled[idx].full_logprobs,
                action_tokens=sampled[idx].action_tokens,
                action_logprobs=sampled[idx].action_logprobs,
                action_length=sampled[idx].action_length,
                generated_text=sampled[idx].text,
                vision=sampled[idx].vision,
                grid=sampled[idx].grid,
                reward=float(rewards[idx]) if idx < len(rewards) else 0.0,
                done=bool(dones[idx]) if idx < len(dones) else True,
                info=info_per_episode[idx] if idx < len(info_per_episode) else {},
            )
        )

    return EpisodeBatch(episodes=episodes, raw_info=info)


def episodes_to_training_batch(
    model: Qwen3VLModel,
    episodes: Sequence[Episode],
    *,
    pad_id: int,
    max_sequence_length: Optional[int],
) -> Tuple[Rollout, Batch]:
    """Convert sampled episodes into Rollout + Batch structures for PPO updates."""
    if not episodes:
        raise ValueError("episodes_to_training_batch requires at least one episode.")

    token_list = [np.asarray(ep.full_tokens, dtype=np.int32) for ep in episodes]
    logprob_list = [np.asarray(ep.full_logprobs, dtype=np.float32) for ep in episodes]
    prompt_lens = np.asarray([ep.prompt_length for ep in episodes], dtype=np.int32)
    action_lens = np.asarray([ep.action_length for ep in episodes], dtype=np.int32)
    tokens = pad_sequences(token_list, pad_id, max_sequence_length)
    old_logprobs = pad_float_sequences(logprob_list, max_len=max_sequence_length)
    mask_targets = build_masks(prompt_lens, action_lens, int(tokens.shape[1]))
    token_mask = (tokens[:, :-1] != pad_id).astype(jnp.int32)
    vision_batch = pad_vision([ep.vision for ep in episodes])
    grid_batch = pad_grid([np.asarray(ep.grid, dtype=np.int32) for ep in episodes])
    cos, sin = build_rope(model, tokens, grid_batch, token_mask)
    returns = jnp.asarray([ep.reward for ep in episodes], dtype=jnp.float32)
    texts = [ep.generated_text for ep in episodes]

    rollout = Rollout(
        tokens=tokens,
        old_logprobs=old_logprobs,
        returns=returns,
        mask_targets=mask_targets,
        prompt_lens=jnp.asarray(prompt_lens, dtype=jnp.int32),
        action_lens=jnp.asarray(action_lens, dtype=jnp.int32),
        texts=texts,
    )
    batch = Batch(
        tokens=tokens,
        token_mask=token_mask,
        cos=cos,
        sin=sin,
        vision=vision_batch,
        grid=grid_batch,
    )
    return rollout, batch


def _resolve_image_pad_id(tokenizer, model_dir: str) -> int:
    """Resolve the <image> placeholder token id with local fallbacks."""
    image_pad = getattr(tokenizer, "image_token_id", None)
    if image_pad is not None and int(image_pad) >= 0:
        return int(image_pad)
    special = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special, dict):
        image_pad = special.get("image_token_id")
        if image_pad is not None and int(image_pad) >= 0:
            return int(image_pad)
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            image_pad = cfg.get("image_token_id", 151655)
            return int(image_pad)
        except Exception:
            pass
    return 151655


def main() -> None:
    """Simple CLI for sampling environment episodes without PPO."""
    parser = argparse.ArgumentParser(description="Collect RL episodes using the unified sampler.")

    # Platform-aware defaults for macOS local runs
    mac = is_macos()
    default_top_k = 256 if mac else None
    default_max_new_tokens = 32 if mac else 64
    default_max_pixels = 16_384 if mac else -1
    default_dtype = "float32" if mac else None
    default_decode_impl = "step" if mac else "scan"
    default_max_seq_len = 256 if mac else None

    parser.add_argument("--model_dir", "--model-dir", required=True, type=str, help="Model directory containing weights, config, and tokenizer.")
    parser.add_argument("--env_name", "--env-name", required=True, type=str, help="Environment identifier (e.g., vision, geospot).")
    parser.add_argument("--episodes", type=int, default=4, help="Total number of episodes to sample.")
    parser.add_argument("--batch_size", type=int, default=1, help="Episodes to sample in parallel.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", "--top-p", type=float, default=0.9)
    parser.add_argument("--top_k", "--top-k", type=int, default=default_top_k)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("--max_sequence_length", "--max-sequence-length", type=int, default=default_max_seq_len, help="Optional cap on total prompt+generation token length.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_pixels", "--min-pixels", type=int, default=-1, help="Override minimum resized pixels (<=0 keeps default).")
    parser.add_argument("--max_pixels", "--max-pixels", type=int, default=default_max_pixels, help="Override maximum resized pixels (<=0 keeps default).")
    parser.add_argument(
        "--dtype",
        type=str,
        default=default_dtype,
        choices=[None, "float32", "bfloat16", "bf16", "fp32"],
        help="Override model compute dtype (use float32 on Metal for stability)",
    )
    parser.add_argument(
        "--decode_impl", "--decode-impl",
        type=str,
        default=default_decode_impl,
        choices=["scan", "step"],
        help="Decode implementation: scan (fast) or step (Metal-safe).",
    )
    parser.add_argument(
        "--decode_unroll", "--decode-unroll",
        type=int,
        default=1,
        help="Unroll factor for scan decode (suggest 4 or 8 on GPU)",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Enable live TUI visualization of attention, KV cache, and sampling",
    )
    parser.add_argument(
        "--tui_mode", "--tui-mode",
        type=str,
        choices=["simple", "advanced", "merged"],
        default="simple",
        help="TUI layout: simple episode list, advanced metrics, or both (merged)",
    )
    args = parser.parse_args()

    # Prefer the most precise matmul on METAL to reduce numerical drift.
    try:
        from jax import config as jax_config

        if jax.devices() and jax.devices()[0].platform.lower() == "metal":
            jax_config.update("jax_default_matmul_precision", "highest")
    except Exception:
        pass

    model, params = create_model_from_ckpt(args.model_dir, dtype=args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    env = create_env(args.env_name, tokenizer)
    image_pad_id = _resolve_image_pad_id(tokenizer, args.model_dir)

    sampling_cfg = SamplingConfig(
        temperature=float(args.temperature),
        top_p=(float(args.top_p) if 0.0 < float(args.top_p) < 1.0 else None),
        top_k=(int(args.top_k) if args.top_k is not None and int(args.top_k) > 0 else None),
        eos_id=getattr(tokenizer, "eos_token_id", None),
        pad_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
        max_new_tokens=int(args.max_new_tokens),
    )

    # Initialize TUI if requested
    tui = None
    if args.tui:
        try:
            from vlmrl.utils.tui_live import LiveTUI, DecodeState, TrainingState, EpisodeView
            tui = LiveTUI(
                num_layers=model.spec.text.num_hidden_layers,
                num_heads=model.spec.text.num_attention_heads,
                display_layers=6,
                max_cache=2048,
                view_mode=args.tui_mode,
            )
            tui.start()
            print("TUI visualization enabled.")
        except Exception as e:
            print(f"Warning: Could not initialize TUI: {e}")
            tui = None

    rng = jax.random.PRNGKey(int(args.seed))
    remaining = max(0, int(args.episodes))
    batch_size = max(1, int(args.batch_size))
    taken = 0
    rewards: List[float] = []

    try:
        while remaining > 0:
            take = min(batch_size, remaining)
            rng, subkey = jax.random.split(rng)
            # Optional per-token decode callback for TUI in step mode
            step_cb = None
            if tui is not None and args.decode_impl == "step":
                def _step_cb(info):
                    try:
                        token_id = int(info.get("token_id", 0))
                        token_text = tokenizer.decode([token_id])
                        prob = float(np.exp(info.get("logprob", 0.0)))
                        cache_len = int(info.get("cache_len", 0))
                        cache_max = int(info.get("cache_max", 0))
                        # mRoPE positions for text tokens share axes; derive from cache len + rope delta
                        rope_delta = int(info.get("rope_delta", 0))
                        mpos = cache_len + rope_delta
                        attn_heads = info.get("attn_heads", None)
                        attn_pattern = None
                        active_head = 0
                        if attn_heads is not None:
                            try:
                                # attn_heads: [N, H] for last N layers
                                attn_np = np.asarray(attn_heads, dtype=np.float32)
                                if attn_np.ndim == 2 and attn_np.shape[1] == model.spec.text.num_attention_heads:
                                    attn_pattern = attn_np
                                    # Choose strongest head from the most recent layer
                                    last_row = attn_np[-1]
                                    active_head = int(np.argmax(last_row))
                            except Exception:
                                attn_pattern = None
                                active_head = 0
                        ds = DecodeState(
                            step=int(info.get("step", 0)),
                            token_id=token_id,
                            token_text=token_text,
                            token_prob=prob,
                            cache_len=cache_len,
                            cache_max=cache_max if cache_max > 0 else 2048,
                            active_layer=max(0, model.spec.text.num_hidden_layers - 1),
                            active_head=int(active_head),
                            mrope_t=mpos,
                            mrope_h=mpos,
                            mrope_w=mpos,
                            attention_pattern=attn_pattern,
                        )
                        tui.update_decode(ds)
                    except Exception:
                        pass
                step_cb = _step_cb

            batch = collect_episodes(
                env,
                tokenizer,
                model,
                params,
                sampling_cfg,
                image_pad_id=image_pad_id,
                batch_size=take,
                rng=subkey,
                min_pixels=(args.min_pixels if args.min_pixels and args.min_pixels > 0 else None),
                max_pixels=(args.max_pixels if args.max_pixels and args.max_pixels > 0 else None),
                max_sequence_length=(args.max_sequence_length if args.max_sequence_length and args.max_sequence_length > 0 else None),
                return_logprobs=bool(tui is not None),
                decode_impl=args.decode_impl,
                decode_unroll=int(max(1, args.decode_unroll)),
                step_callback=step_cb,
            )
            for episode in batch.episodes:
                rewards.append(float(episode.reward))
                response_preview = episode.generated_text.replace("\n", " ").strip()
                prompt_preview = episode.prompt_text.strip().splitlines()[-1] if episode.prompt_text.strip() else ""
                info_keys = ", ".join(sorted(episode.info.keys())) if episode.info else "none"

                # Update TUI with episode data
                if tui is not None:
                    # Training state update
                    training_state = TrainingState(
                        step=taken,
                        reward_mean=float(np.mean(rewards[-10:])) if rewards else 0.0,
                        reward_std=float(np.std(rewards[-10:])) if len(rewards) > 1 else 0.0,
                        kl=0.0,  # Not available in rollout-only mode
                        loss=0.0,
                        tokens_per_sec=0.0,  # Could calculate from timing
                        gpu_util=0.0,
                    )
                    tui.update_training(training_state)

                    # Token credit update
                    tokens_text = [tokenizer.decode([t]) for t in episode.action_tokens[:10]]  # First 10 tokens
                    # Convert log-probs to probabilities for readable bars
                    credits_lp = np.asarray(episode.action_logprobs[:10], dtype=np.float32)
                    if credits_lp.size > 0:
                        credits = np.clip(np.exp(credits_lp), 0.0, 1.0)
                        tui.update_credit(tokens_text, credits.tolist())

                    # Simple decode: append full prompt/output for this episode
                    try:
                        ev = EpisodeView(
                            idx=taken,
                            reward=float(episode.reward),
                            done=bool(episode.done),
                            prompt=str(episode.prompt_text or ""),
                            output=str(episode.generated_text or ""),
                        )
                        tui.update_episode(ev)
                    except Exception:
                        pass

                # Print to console (unless TUI is active)
                if tui is None:
                    print(
                        f"[episode {taken}] reward={episode.reward:.3f} done={episode.done} "
                        f"prompt='{prompt_preview}' response='{response_preview}' info_keys={info_keys}"
                    )
                taken += 1
            remaining -= take

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        # Cleanup TUI
        if tui is not None:
            tui.stop()

    if rewards:
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        print(f"Collected {len(rewards)} episodes. reward_mean={mean_reward:.3f} reward_std={std_reward:.3f}")
    else:
        print("No episodes collected.")


if __name__ == "__main__":
    main()


__all__ = [
    "Episode",
    "EpisodeBatch",
    "collect_episodes",
    "episodes_to_training_batch",
]
