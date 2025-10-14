"""
Minimal on-policy GRPO loop for Qwen2.5-VL vision-language training.

The goal is to keep the reinforcement-learning core easy to follow: collect
rollouts, compute a mean-baseline advantage, and apply REINFORCE with optional
entropy regularization. Vision-specific plumbing (encoding images, building
mixed RoPE, etc.) is preserved so this works with existing VLM environments.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import ml_collections
from absl import app, flags

from transformers import AutoTokenizer

from vlmrl.models.qwen25vl import (
    Qwen25VLModel,
    create_model_from_ckpt,
    build_mrope,
    get_rope_index,
)
from vlmrl.utils.configs import define_flag_dict
from vlmrl.utils.wandb import setup_wandb
from vlmrl.envs.env_creator import create_env
from vlmrl.utils.sharding import create_sharding, host_gather
from vlmrl.utils.train_state import TrainState
from vlmrl.utils.checkpoint import Checkpoint
from vlmrl.core.sampling import Sampler as VLMSampler, _resolve_image_pad_id
from vlmrl.utils.vlm import (
    preprocess_image,
    chat_prompt_with_image,
    chat_prompt_with_images,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MAX_PIXELS,
)


config = ml_collections.ConfigDict({
    "wandb_project": "lmpo",
    "wandb_name": "grpo-simple",
    "wandb_group": "Default",
    "model_dir": "checkpoints/qwen25vl_window",
    "save_dir": "",
    "save_interval": 200,
    "env_name": "vision",
    "total_steps": 1000,
    "num_generation_tokens": 64,
    "inference_batch_per_device": 1,
    "groups_per_batch": 8,
    "lr": 5e-7,
    "entropy_coef": 0.0,
    "kl_coef": 0.0,
    "weight_decay": 1e-2,
    "optimizer": "adafactor",
    "vlm_min_pixels": -1,
    "vlm_max_pixels": -1,
    "temperature": 0.7,
    "top_k": 1024,
    "top_p": 0.9,
    "test_env_name": "",
    "test_interval": 200,
    "gradient_checkpointing": False,
    "max_sequence_length": 2048,
})
define_flag_dict(config)
FLAGS = flags.FLAGS


def _pad_sequences(seqs: List[np.ndarray], pad_val: int, max_len: Optional[int] = None) -> jnp.ndarray:
    if not seqs:
        return jnp.zeros((0, 0), dtype=jnp.int32)
    actual_max = max(int(seq.shape[0]) for seq in seqs)
    target_len = min(actual_max, max_len) if max_len else actual_max
    out = np.full((len(seqs), target_len), pad_val, dtype=np.int32)
    for i, seq in enumerate(seqs):
        arr = np.asarray(seq, dtype=np.int32)
        arr = arr[:target_len]
        out[i, :arr.shape[0]] = arr
    return jnp.asarray(out)


def _pad_float_sequences(
    seqs: List[np.ndarray],
    pad_val: float = 0.0,
    max_len: Optional[int] = None,
) -> jnp.ndarray:
    if not seqs:
        return jnp.zeros((0, 0), dtype=jnp.float32)
    actual_max = max(int(seq.shape[0]) for seq in seqs)
    target_len = min(actual_max, max_len) if max_len else actual_max
    out = np.full((len(seqs), target_len), pad_val, dtype=np.float32)
    for i, seq in enumerate(seqs):
        arr = np.asarray(seq, dtype=np.float32)
        arr = arr[:target_len]
        out[i, :arr.shape[0]] = arr
    return jnp.asarray(out)


def _pad_vision(arrs: List[np.ndarray]) -> jnp.ndarray:
    if not arrs:
        return jnp.zeros((0, 0, 0), dtype=jnp.float32)
    dim = int(arrs[0].shape[-1])
    max_tokens = max(int(arr.shape[0]) for arr in arrs)
    out = np.zeros((len(arrs), max_tokens, dim), dtype=np.float32)
    for i, arr in enumerate(arrs):
        out[i, :arr.shape[0], :] = np.asarray(arr, dtype=np.float32)
    return jnp.asarray(out)


def _pad_grid(arrs: List[np.ndarray]) -> jnp.ndarray:
    if not arrs:
        return jnp.zeros((0, 0, 3), dtype=jnp.int32)
    max_rows = max(int(arr.shape[0]) for arr in arrs)
    out = np.zeros((len(arrs), max_rows, 3), dtype=np.int32)
    for i, arr in enumerate(arrs):
        np_arr = np.asarray(arr, dtype=np.int32)
        out[i, :np_arr.shape[0], :] = np_arr
    return jnp.asarray(out)


def _prepare_prompt_and_embeds(
    model: Qwen25VLModel,
    train_state: TrainState,
    tokenizer,
    obs,
    vlm_min_pixels: int | None,
    vlm_max_pixels: int | None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    vision_spec = model.spec.vision
    if vision_spec is None:
        raise ValueError("Model must include a vision backbone")

    def _pixels(val):
        if val and val > 0:
            return val
        return None

    min_pixels = _pixels(vlm_min_pixels) or DEFAULT_MIN_PIXELS
    max_pixels = _pixels(vlm_max_pixels) or DEFAULT_MAX_PIXELS

    if hasattr(obs, "image_path") and hasattr(obs, "question"):
        pix, grid = preprocess_image(
            obs.image_path,
            patch_size=vision_spec.patch_size,
            spatial_merge_size=vision_spec.spatial_merge_size,
            temporal_patch_size=vision_spec.temporal_patch_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        embeds = model.apply(
            {"params": train_state.params},
            pix,
            grid,
            method=model.encode_vision,
        )
        num_tokens = int(embeds.shape[0])
        prompt_text = chat_prompt_with_image(num_tokens, getattr(obs, "question", ""))
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)
        return prompt_tokens, embeds, grid

    if (
        hasattr(obs, "image_left")
        and hasattr(obs, "image_right")
        and hasattr(obs, "statement")
    ):
        pix_l, grid_l = preprocess_image(
            obs.image_left,
            patch_size=vision_spec.patch_size,
            spatial_merge_size=vision_spec.spatial_merge_size,
            temporal_patch_size=vision_spec.temporal_patch_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        pix_r, grid_r = preprocess_image(
            obs.image_right,
            patch_size=vision_spec.patch_size,
            spatial_merge_size=vision_spec.spatial_merge_size,
            temporal_patch_size=vision_spec.temporal_patch_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        emb_l = model.apply(
            {"params": train_state.params},
            pix_l,
            grid_l,
            method=model.encode_vision,
        )
        emb_r = model.apply(
            {"params": train_state.params},
            pix_r,
            grid_r,
            method=model.encode_vision,
        )
        embeds = jnp.concatenate([emb_l, emb_r], axis=0)
        grid = jnp.concatenate([grid_l, grid_r], axis=0)
        prompt_text = chat_prompt_with_images(
            [int(emb_l.shape[0]), int(emb_r.shape[0])],
            f"Look at the two images. {getattr(obs, 'statement', '')}",
        )
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)
        return prompt_tokens, embeds, grid

    raise ValueError("Unsupported observation structure for vision RL")


def _build_masks(prompt_lens: np.ndarray, action_lens: np.ndarray, seq_len: int) -> jnp.ndarray:
    mask = np.zeros((len(prompt_lens), seq_len), dtype=np.int32)
    for i, (prompt_len, action_len) in enumerate(zip(prompt_lens, action_lens, strict=True)):
        start = max(int(prompt_len) - 1, 0)
        end = min(start + max(int(action_len) - 1, 0), seq_len - 1)
        if end >= start:
            mask[i, start : end + 1] = 1
    return jnp.asarray(mask)


def collect_rollouts(
    train_state: TrainState,
    sampler: VLMSampler,
    env,
    tokenizer,
    model: Qwen25VLModel,
    pad_id: int,
    image_pad_id: int,
    rng: jax.random.PRNGKey,
    shard_data_fn,
    rollout_batch_size: int,
    groups_per_batch: int,
    env_task_idx: int,
    vlm_min_pixels: int | None,
    vlm_max_pixels: int | None,
    eos_id: int | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    max_seq_len: Optional[int] = None,
) -> Tuple[jax.random.PRNGKey, int, Dict[str, jnp.ndarray], Dict[str, float]]:
    buffer_tokens: List[jnp.ndarray] = []
    buffer_logprobs: List[jnp.ndarray] = []
    buffer_rewards: List[float] = []
    buffer_prompt_lens: List[int] = []
    buffer_action_lens: List[int] = []
    buffer_vis_embeds: List[jnp.ndarray] = []
    buffer_grid: List[jnp.ndarray] = []
    buffer_texts: List[str] = []
    info_sums: Dict[str, float] = {}
    info_counts: Dict[str, int] = {}

    env_num_tasks = env.num_tasks if env.num_tasks != -1 else 1_000_000
    stats = {"rollouts": 0.0, "reward_mean": 0.0}

    while len(buffer_tokens) < groups_per_batch:
        states = []
        observations = []
        for _ in range(rollout_batch_size):
            idx = min(env_task_idx + jax.process_index(), env_num_tasks - 1)
            env_state, obs = env.reset(idx)
            env_task_idx = (env_task_idx + jax.process_count()) % env_num_tasks
            states.append(env_state)
            observations.append(obs)

        actions = []
        tokens_local = []
        logprobs_local = []
        prompt_lens_local = []
        action_lens_local = []
        embeds_local = []
        grid_local = []
        responses_local = []

        for obs in observations:
            prompt_tokens, embeds, grid = _prepare_prompt_and_embeds(
                model,
                train_state,
                tokenizer,
                obs,
                vlm_min_pixels,
                vlm_max_pixels,
            )
            prompt_len = int(prompt_tokens.shape[1])
            rng, key = jax.random.split(rng)
            generated, sample_logprobs = sampler.sample_vlm(
                prompt_tokens,
                embeds,
                grid,
                image_pad_id=image_pad_id,
                max_new_tokens=int(FLAGS.num_generation_tokens),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_id=eos_id,
                pad_id=pad_id,
                rng=key,
                return_logprobs=True,
            )
            gen_tokens = generated[0]
            gen_logprobs = sample_logprobs[0]
            prompt_arr = prompt_tokens[0]
            full_tokens = jnp.concatenate([prompt_arr, gen_tokens], axis=0)

            prompt_len_effective = int(prompt_len)
            if max_seq_len is not None:
                max_tokens_allowed = int(max_seq_len)
                if full_tokens.shape[0] > max_tokens_allowed:
                    allowed_actions = max_tokens_allowed - prompt_len
                    prompt_slice_len = max(0, min(prompt_len, max_tokens_allowed))
                    prompt_arr = prompt_arr[:prompt_slice_len]
                    if allowed_actions <= 0:
                        gen_tokens = gen_tokens[:0]
                        gen_logprobs = gen_logprobs[:0]
                    else:
                        gen_tokens = gen_tokens[:allowed_actions]
                        gen_logprobs = gen_logprobs[:allowed_actions]
                    full_tokens = jnp.concatenate([prompt_arr, gen_tokens], axis=0)
                    prompt_len_effective = int(prompt_arr.shape[0])
            zeros = jnp.zeros((prompt_len_effective,), dtype=jnp.float32)
            full_logprobs = jnp.concatenate([zeros, gen_logprobs], axis=0)

            actions.append(gen_tokens.tolist())
            tokens_local.append(full_tokens)
            logprobs_local.append(full_logprobs)
            prompt_lens_local.append(prompt_len_effective)
            action_lens_local.append(int(gen_tokens.shape[0]))
            embeds_local.append(embeds)
            grid_local.append(grid)
            decoded = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
            responses_local.append(decoded)

        _, _, returns_local, _, infos_local = env.step_list(states, actions)
        returns_np = np.asarray(returns_local, dtype=np.float32)
        returns_global = host_gather(shard_data_fn(returns_np))

        if infos_local:
            for key, values in infos_local.items():
                if values is None:
                    continue
                arr = np.asarray(values)
                if arr.size == 0 or arr.dtype.kind not in {"b", "i", "u", "f"}:
                    continue
                arr = arr.astype(np.float32, copy=False)
                gathered = host_gather(shard_data_fn(arr))
                gathered_np = np.asarray(gathered, dtype=np.float32)
                info_sums[key] = info_sums.get(key, 0.0) + float(gathered_np.sum())
                info_counts[key] = info_counts.get(key, 0) + int(gathered_np.size)

        stats["rollouts"] += float(len(returns_global))
        if len(buffer_rewards) + len(returns_global) > 0:
            total_rewards = buffer_rewards + returns_global.tolist()
            stats["reward_mean"] = float(np.mean(total_rewards))

        for token_arr, logprob_arr, rew, prompt_len, action_len, embeds, grid, text in zip(
            tokens_local,
            logprobs_local,
            returns_global.tolist(),
            prompt_lens_local,
            action_lens_local,
            embeds_local,
            grid_local,
            responses_local,
            strict=True,
        ):
            if len(buffer_tokens) >= groups_per_batch:
                break
            buffer_tokens.append(token_arr)
            buffer_logprobs.append(logprob_arr)
            buffer_rewards.append(float(rew))
            buffer_prompt_lens.append(int(prompt_len))
            buffer_action_lens.append(int(action_len))
            buffer_vis_embeds.append(embeds)
            buffer_grid.append(grid)
            buffer_texts.append(text)

    tokens = _pad_sequences(buffer_tokens, pad_id, max_seq_len)
    sample_logprobs = _pad_float_sequences(buffer_logprobs, max_len=max_seq_len)
    prompt_lens_arr = np.asarray(buffer_prompt_lens, dtype=np.int32)
    action_lens_arr = np.asarray(buffer_action_lens, dtype=np.int32)
    mask = _build_masks(prompt_lens_arr, action_lens_arr, int(tokens.shape[1]))

    vision_embeds = _pad_vision([np.asarray(v) for v in buffer_vis_embeds])
    grid = _pad_grid([np.asarray(g) for g in buffer_grid])

    rewards = jnp.asarray(buffer_rewards, dtype=jnp.float32)
    advantages = rewards - rewards.mean()

    batch = {
        "tokens": tokens,
        "mask": mask,
        "sample_logprobs": sample_logprobs,
        "advantages": advantages,
        "rewards": rewards,
        "vision_embeds": vision_embeds,
        "grid": grid,
        "responses": buffer_texts,
    }

    if info_counts:
        env_metrics = {}
        for key, total in info_sums.items():
            count = info_counts.get(key, 0)
            if count > 0:
                env_metrics[key] = float(total / count)
        if env_metrics:
            stats["env_metrics"] = env_metrics

    return rng, env_task_idx, batch, stats


def compute_rope_indices_chunked(
    model: Qwen25VLModel,
    text_input: jnp.ndarray,
    grid: jnp.ndarray,
    token_mask: jnp.ndarray,
    max_chunk_size: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    batch_size = int(text_input.shape[0])
    if batch_size > max_chunk_size:
        cos_chunks = []
        sin_chunks = []
        for start in range(0, batch_size, max_chunk_size):
            end = min(start + max_chunk_size, batch_size)
            chunk_input = text_input[start:end]
            chunk_grid = grid[start:end]
            chunk_mask = token_mask[start:end]
            pos3_chunk, _ = get_rope_index(
                spatial_merge_size=model.spec.vision.spatial_merge_size,
                input_ids=chunk_input,
                image_grid_thw=chunk_grid,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=chunk_mask,
                tokens_per_second=float(model.spec.vision.tokens_per_second),
            )
            cos_chunk, sin_chunk = build_mrope(
                pos3_chunk,
                tuple(model.spec.text.rope_section),
                float(model.spec.text.rope_theta),
                dtype=model.dtype,
                rope_scaling_type=getattr(model.spec.text, "rope_scaling_type", None),
                rope_scaling_factor=getattr(model.spec.text, "rope_scaling_factor", None),
            )
            cos_chunks.append(cos_chunk)
            sin_chunks.append(sin_chunk)
        cos = jnp.concatenate(cos_chunks, axis=1)
        sin = jnp.concatenate(sin_chunks, axis=1)
    else:
        pos3, _ = get_rope_index(
            spatial_merge_size=model.spec.vision.spatial_merge_size,
            input_ids=text_input,
            image_grid_thw=grid,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=token_mask,
            tokens_per_second=float(model.spec.vision.tokens_per_second),
        )
        cos, sin = build_mrope(
            pos3,
            tuple(model.spec.text.rope_section),
            float(model.spec.text.rope_theta),
            dtype=model.dtype,
            rope_scaling_type=getattr(model.spec.text, "rope_scaling_type", None),
            rope_scaling_factor=getattr(model.spec.text, "rope_scaling_factor", None),
        )
    return cos, sin


def main(_):
    FLAGS(sys.argv)
    if jax.process_index() == 0:
        setup_wandb(
            FLAGS.flag_values_dict(),
            project=FLAGS.wandb_project,
            name=f"{FLAGS.env_name}-{FLAGS.wandb_name}",
            group=FLAGS.wandb_group,
        )

    model, params = create_model_from_ckpt(FLAGS.model_dir)
    optimizer_name = str(getattr(FLAGS, "optimizer", "adafactor")).lower()
    if optimizer_name == "adamw":
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=FLAGS.lr,
                b1=0.9,
                b2=0.95,
                weight_decay=FLAGS.weight_decay,
            ),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adafactor(
                learning_rate=FLAGS.lr,
                min_dim_size_to_factor=32,
                dtype_momentum=jnp.bfloat16,
                weight_decay_rate=(
                    float(FLAGS.weight_decay) if float(FLAGS.weight_decay) > 0 else None
                ),
            ),
        )
    rng = jax.random.PRNGKey(0)
    init_fn = lambda rng: TrainState.create_with_params(
        model_def=model,
        tx=tx,
        params=params,
        rng=rng,
        use_ema=False,
    )
    train_state = init_fn(rng)

    shard_mode = str(getattr(FLAGS, "shard", "dp")).lower()
    train_state_shape = jax.eval_shape(lambda ts: ts, train_state)
    _train_state_shard, no_shard, data_shard, shard_data_fn = create_sharding(
        shard_mode, train_state_shape
    )
    if shard_mode == "dp":
        train_state = jax.device_put(train_state, no_shard)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_dir, trust_remote_code=False)
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    eos_id = getattr(tokenizer, "eos_token_id", None)

    env = create_env(FLAGS.env_name, tokenizer)
    env_test = create_env(FLAGS.test_env_name, tokenizer) if FLAGS.test_env_name else None

    sampler = VLMSampler(model, train_state.params)
    image_pad_id = _resolve_image_pad_id(tokenizer, FLAGS.model_dir)

    vlm_min_pixels = FLAGS.vlm_min_pixels if hasattr(FLAGS, "vlm_min_pixels") else -1
    vlm_max_pixels = FLAGS.vlm_max_pixels if hasattr(FLAGS, "vlm_max_pixels") else -1
    vlm_min_pixels = None if int(vlm_min_pixels) <= 0 else int(vlm_min_pixels)
    vlm_max_pixels = None if int(vlm_max_pixels) <= 0 else int(vlm_max_pixels)

    top_k = int(FLAGS.top_k) if int(getattr(FLAGS, "top_k", -1) or -1) > 0 else None
    top_p = float(FLAGS.top_p)
    top_p = top_p if (0.0 < top_p < 1.0) else None
    temperature = float(getattr(FLAGS, "temperature", 1.0) or 1.0)

    max_seq_len = int(getattr(FLAGS, "max_sequence_length", 0) or 0)
    if max_seq_len <= 0:
        max_seq_len = None

    rollout_batch_size = jax.local_device_count() * int(FLAGS.inference_batch_per_device)
    rng_global = jax.random.PRNGKey(jax.process_index())
    env_task_idx = 0

    @jax.jit
    def _update(train_state, tokens, mask, advantages, sample_logprobs, vision_embeds, token_mask, cos, sin):
        text_input = tokens[:, :-1]
        text_target = tokens[:, 1:]
        mask_tokens = mask[:, 1:]

        def loss_fn(params):
            def call_with_params(p):
                return train_state.call_model(
                    text_input,
                    vision_embeds,
                    image_pad_id,
                    cos,
                    sin,
                    mask=token_mask,
                    cache=None,
                    params=p,
                    method=model.forward_vlm,
                )

            if getattr(FLAGS, 'gradient_checkpointing', False):
                logits, _ = jax.checkpoint(call_with_params, prevent_cse=False)(params)
            else:
                logits, _ = call_with_params(params)
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            token_logprobs = jnp.sum(
                logprobs * jax.nn.one_hot(text_target, logits.shape[-1]),
                axis=-1,
            )
            entropy = -jnp.sum(jax.nn.softmax(logits) * logprobs, axis=-1)

            adv = advantages[:, None]
            mask_total = jnp.sum(mask_tokens) + 1e-8
            pg_loss = -jnp.sum(token_logprobs * adv * mask_tokens) / mask_total
            entropy_loss = -FLAGS.entropy_coef * jnp.sum(entropy * mask_tokens) / mask_total

            if FLAGS.kl_coef and float(FLAGS.kl_coef) > 0:
                old_logprobs = sample_logprobs[:, 1:]
                kl_term = jnp.sum((token_logprobs - old_logprobs) * mask_tokens) / mask_total
                kl_penalty = float(FLAGS.kl_coef) * kl_term
            else:
                kl_term = 0.0
                kl_penalty = 0.0

            loss = pg_loss + entropy_loss + kl_penalty
            metrics = {
                "loss": loss,
                "loss_pg": pg_loss,
                "loss_entropy": entropy_loss,
                "kl_term": kl_term,
                "mask_tokens": mask_total,
                "entropy": jnp.sum(entropy * mask_tokens) / mask_total,
                "token_logprob_mean": jnp.sum(token_logprobs * mask_tokens) / mask_total,
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        updates, opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)
        new_state = train_state.replace(
            params=new_params,
            opt_state=opt_state,
            step=train_state.step + 1,
        )
        metrics = {**metrics, "grad_norm": optax.global_norm(grads)}
        return new_state, metrics

    for step_idx in range(int(FLAGS.total_steps)):
        if step_idx == 0 and jax.process_index() == 0:
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")

        sampler.params = train_state.params
        rng_global, env_task_idx, batch, rollout_stats = collect_rollouts(
            train_state,
            sampler,
            env,
            tokenizer,
            model,
            pad_id,
            image_pad_id,
            rng_global,
            shard_data_fn,
            rollout_batch_size,
            int(FLAGS.groups_per_batch),
            env_task_idx,
            vlm_min_pixels,
            vlm_max_pixels,
            eos_id,
            temperature,
            top_k,
            top_p,
            max_seq_len=max_seq_len,
        )

        text_input = batch["tokens"][:, :-1]
        token_mask = (text_input != pad_id).astype(jnp.int32)
        cos, sin = compute_rope_indices_chunked(
            model, text_input, batch["grid"], token_mask
        )

        train_state, metrics = _update(
            train_state,
            batch["tokens"],
            batch["mask"],
            batch["advantages"],
            batch["sample_logprobs"],
            batch["vision_embeds"],
            token_mask,
            cos,
            sin,
        )

        info = {
            "global_step": int(step_idx),
            "reward/mean": float(batch["rewards"].mean()),
            "reward/std": float(batch["rewards"].std()),
            "advantage/std": float(batch["advantages"].std()),
            "rollouts_per_step": rollout_stats["rollouts"],
        }
        info.update({f"train/{k}": float(v) for k, v in metrics.items()})
        env_metrics = rollout_stats.get("env_metrics", {})
        for k, v in env_metrics.items():
            info[k] = float(v)

        if jax.process_index() == 0:
            sample_reward = float(batch["rewards"][0]) if batch["rewards"].size > 0 else float("nan")
            sample_text = batch["responses"][0] if batch["responses"] else ""
            preview = sample_text if len(sample_text) <= 200 else sample_text[:200] + "..."
            print(
                f"[step {step_idx}] reward_mean={info['reward/mean']:.3f} "
                f"reward_sample={sample_reward:.3f} loss={info.get('train/loss', float('nan')):.4f}"
            )
            if sample_text:
                print(f"[step {step_idx}] sample response: {preview}")
            if env_metrics:
                formatted_env = ", ".join(f"{key}={value:.3f}" for key, value in env_metrics.items())
                print(f"[step {step_idx}] env metrics: {formatted_env}")
            wandb.log(info, commit=True)

        if step_idx % int(FLAGS.save_interval) == 0 and FLAGS.save_dir and jax.process_index() == 0:
            params_gather = host_gather(train_state.params)
            ckpt_dir = f"{FLAGS.save_dir}/step{step_idx}/"
            cp = Checkpoint(f"{ckpt_dir}params.pkl", parallel=False)
            cp.params = params_gather
            cp.save()

        if (
            env_test is not None
            and step_idx > 0
            and step_idx % int(FLAGS.test_interval) == 0
            and jax.process_index() == 0
        ):
            states, obs_batch = [], []
            for _ in range(rollout_batch_size):
                idx = min(env_task_idx + jax.process_index(), env.num_tasks - 1)
                state, obs = env_test.reset(idx)
                states.append(state)
                obs_batch.append(obs)
            actions = []
            for obs in obs_batch:
                prompt_tokens, embeds, grid = _prepare_prompt_and_embeds(
                    model,
                    train_state,
                    tokenizer,
                    obs,
                    vlm_min_pixels,
                    vlm_max_pixels,
                )
                rng_global, key = jax.random.split(rng_global)
                generated, _ = sampler.sample_vlm(
                    prompt_tokens,
                    embeds,
                    grid,
                    image_pad_id=image_pad_id,
                    max_new_tokens=int(FLAGS.num_generation_tokens),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    rng=key,
                    return_logprobs=False,
                )
                actions.append(generated[0].tolist())
            _, _, ret_local, _, _ = env_test.step_list(states, actions)
            ret = host_gather(shard_data_fn(np.asarray(ret_local, dtype=np.float32)))
            wandb.log({"test/return": float(np.mean(ret)), "global_step": int(step_idx)}, commit=False)


if __name__ == "__main__":
    app.run(main)
