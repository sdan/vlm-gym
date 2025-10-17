"""
Grouped PPO/GRPO loop for Qwen3-VL vision-language training (efficient variant).

This implements grouped sampling (GRPO), minibatched PPO updates, and rich
metrics while preserving Qwen3-VL vision plumbing (image encoding, mixed RoPE).
"""

from __future__ import annotations

# Set XLA/JAX runtime flags early (before importing jax) to reduce GPU OOM
# and autotuning memory overhead. Users can override via environment.
import os as _os
_os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=3")
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import time
import shutil
from functools import partial
from typing import Dict, List, Tuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import ml_collections
from absl import app, flags

from transformers import AutoTokenizer

from vlmrl.models.qwen3vl.model import (
    VisionEmbeddings,
    create_model_from_ckpt,
)
from vlmrl.utils.configs import define_flag_dict
from vlmrl.utils.wandb import setup_wandb
from vlmrl.envs.env_creator import create_env
from vlmrl.utils.sharding import create_sharding, host_gather
from vlmrl.utils.train_state import TrainState
from vlmrl.utils.checkpoint import Checkpoint
from vlmrl.core.sampling import Sampler as VLMSampler, _resolve_image_pad_id

# Reuse helpers from the minimal GRPO implementation
from vlmrl.core.grpo import (
    _prepare_prompt_and_embeds,
    _pad_sequences,
    _pad_float_sequences,
    _pad_vision,
    _pad_grid,
    _build_masks,
    compute_rope_indices_chunked,
)
import vlmrl.core.grpo as grpo_core


config = ml_collections.ConfigDict({
    # WandB / bookkeeping
    "wandb_project": "lmpo",
    "wandb_name": "grpo-full",
    "wandb_group": "Default",

    # Model / IO
    "model_dir": "checkpoints/qwen3vl",
    "save_dir": "",
    "save_interval": 10,
    "resume_dir": "",
    "resume_step": 0,

    # Env / evaluation
    "env_name": "vision",
    "test_env_name": "",
    "test_interval": 200,
    # OSV5M/Geo env knobs (only used if env_name in {osv5m, geospot})
    "env_split": "test",
    "env_coord_tolerance_km": 25.0,
    "env_use_geo_shaping": 1,
    "env_geo_decay_km": 300.0,
    "env_country_weight": 0.2,
    "env_region_weight": 0.3,
    "env_city_weight": 0.5,
    # Optional dense shaping
    "env_format_bonus_base": 0.05,
    "env_format_bonus_per_field": 0.0,
    "env_presence_bonus_scale": 0.0,
    "env_coords_presence_bonus": 0.0,
    "env_max_samples": -1,  # -1 means all available

    # Sampling
    "num_generation_tokens": 64,
    "temperature": 0.7,
    "top_k": 1024,
    "top_p": 0.9,
    "inference_batch_per_device": 8,
    "prefill_batch_split": 1,  # placeholder; sampler handles batching internally

    # Vision preprocessing overrides (<=0 means use defaults)
    "vlm_min_pixels": -1,
    "vlm_max_pixels": -1,

    # Training duration
    "total_steps": 1000,

    # GRPO / PPO knobs
    "groups_per_batch": 64,
    "group_size": 8,
    "ppo_minibatch": 64,
    "clip_epsilon": 0.2,
    "do_ppo_all_clip": 0,  # 1 to clip both sides directly
    "entropy_coef": 0.001,
    "kl_coef": 0.0,  # optional extra KL regularizer (off by default)
    # Adaptive KL (trust-region style) to prevent early collapse
    "adaptive_kl": 1,            # 1 to enable dynamic KL coefficient
    "kl_target": 0.02,           # desired approx_kl per update
    "kl_adapt_rate": 1.5,        # multiplicative adjust factor
    "kl_coef_min": 1e-5,         # floor and cap to keep sane
    "kl_coef_max": 0.5,
    # Trust-region gradient scaling (stabilize when KL spikes)
    "kl_scale_updates": 1,       # 1 to scale grads when approx_kl > kl_target
    "kl_scale_min": 0.05,        # minimum scale when clamping
    # Entropy stabilization: scale entropy loss down when advantages vanish
    "entropy_scale_by_adv": 1,   # 1 to downscale entropy term if adv std is tiny
    "entropy_target_adv_std": 0.3,  # target adv std for full entropy
    "entropy_scale_min": 0.0,    # min entropy scale

    # Advantage processing
    "do_group_normalization": 1,
    "do_global_normalization": 0,
    "do_group_filter": 1,
    "do_clip_advantages": 0,

    # Importance-ratio and inference/recompute options
    "do_inference_logprobs": 0,
    "do_mask_inference_ratio": 0,
    "do_mask_importance_ratio": 0,

    # Optimizer
    "lr": 1e-6,
    "optimizer": "adamw",
    "weight_decay": 1e-2,
    "update_clip": 0.0,
    "max_grad_norm": 1.0,

    # Model forward
    "gradient_checkpointing": 0,
    "max_sequence_length": 2048,
    # EMA options
    "use_ema": 0,             # 1 to maintain EMA weights
    "ema_tau": 0.999,         # EMA decay (closer to 1 means slower update)
    "sample_with_ema": 1,     # 1 to use EMA params for sampling
    # Memory knobs
    "skip_vision_deepstack": 1,  # drop deepstack features in batches to reduce memory
})
define_flag_dict(config)
FLAGS = flags.FLAGS


def _decode_sample(tokenizer, tokens: jnp.ndarray, prompt_len: int) -> str:
    try:
        gen = np.asarray(tokens[prompt_len:], dtype=np.int32).tolist()
        return tokenizer.decode(gen, skip_special_tokens=True)
    except Exception:
        return ""


@partial(jax.jit, donate_argnums=(2, 3, 4, 5, 6))
def _compute_token_logprobs(train_state: TrainState,
                            image_pad_id: int,
                            tokens: jnp.ndarray,
                            token_mask: jnp.ndarray,
                            vision_embeds: Union[jnp.ndarray, VisionEmbeddings],
                            cos: jnp.ndarray,
                            sin: jnp.ndarray) -> jnp.ndarray:
    text_input = tokens[:, :-1]
    text_target = tokens[:, 1:]
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
            method=train_state.model_def.forward_vlm,
        )
    if getattr(FLAGS, 'gradient_checkpointing', False):
        logits, _ = jax.checkpoint(call_with_params, prevent_cse=False)(train_state.params)
    else:
        logits, _ = call_with_params(train_state.params)
    # Memory-efficient token logprobs: gather logits and subtract logsumexp
    lse = jax.scipy.special.logsumexp(logits, axis=-1)  # [B, T-1]
    gathered = jnp.take_along_axis(logits, text_target[..., None], axis=-1)[..., 0]
    token_logp = gathered - lse
    return token_logp


def main(_):
    FLAGS(sys.argv)
    # Optional: reduce memory by skipping deepstack copying in _pad_vision
    try:
        grpo_core.set_skip_deepstack(bool(int(getattr(FLAGS, "skip_vision_deepstack", 1) or 0) == 1))
    except Exception:
        pass
    if jax.process_index() == 0:
        setup_wandb(
            FLAGS.flag_values_dict(),
            project=FLAGS.wandb_project,
            name=f"{FLAGS.env_name}-{FLAGS.wandb_name}",
            group=FLAGS.wandb_group,
        )

    # Model & optimizer
    model, params = create_model_from_ckpt(FLAGS.model_dir)
    clip_fns: List[optax.GradientTransformation] = []
    if float(getattr(FLAGS, "update_clip", 0.0) or 0.0) > 0:
        clip_fns.append(optax.clip(float(FLAGS.update_clip)))
    if float(getattr(FLAGS, "max_grad_norm", 0.0) or 0.0) > 0:
        clip_fns.append(optax.clip_by_global_norm(float(FLAGS.max_grad_norm)))
    optimizer_name = str(getattr(FLAGS, "optimizer", "adamw")).lower()
    if optimizer_name == "adamw":
        base_opt = optax.adamw(
            learning_rate=FLAGS.lr,
            b1=0.9,
            b2=0.95,
            weight_decay=FLAGS.weight_decay,
        )
    else:
        base_opt = optax.adafactor(
            learning_rate=FLAGS.lr,
            min_dim_size_to_factor=32,
            dtype_momentum=jnp.bfloat16,
            weight_decay_rate=(float(FLAGS.weight_decay) if float(FLAGS.weight_decay) > 0 else None),
        )
    tx = optax.chain(*clip_fns, base_opt) if clip_fns else base_opt

    rng = jax.random.PRNGKey(0)
    use_ema_flag = bool(int(getattr(FLAGS, "use_ema", 0) or 0) == 1)
    init_fn = lambda rng: TrainState.create_with_params(
        model_def=model,
        tx=tx,
        params=params,
        rng=rng,
        use_ema=use_ema_flag,
    )
    train_state = init_fn(rng)

    shard_mode = str(getattr(FLAGS, "shard", "dp")).lower()
    train_state_shape = jax.eval_shape(lambda ts: ts, train_state)
    _train_state_shard, no_shard, data_shard, shard_data_fn = create_sharding(
        shard_mode, train_state_shape
    )
    if shard_mode == "dp":
        train_state = jax.device_put(train_state, no_shard)

    # Tokenizer & env
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_dir, trust_remote_code=False)
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Thread environment kwargs for OSV5M-style envs
    def _mk_env(name: str):
        name = str(name or "").strip()
        if not name:
            return None
        if name.lower() in {"osv5m", "geospot"}:
            max_samples = int(getattr(FLAGS, "env_max_samples", -1) or -1)
            kwargs = dict(
                split=str(getattr(FLAGS, "env_split", "test") or "test"),
                coord_tolerance_km=float(getattr(FLAGS, "env_coord_tolerance_km", 25.0) or 25.0),
                use_geo_shaping=bool(int(getattr(FLAGS, "env_use_geo_shaping", 1) or 1)),
                geo_decay_km=float(getattr(FLAGS, "env_geo_decay_km", 300.0) or 300.0),
                country_weight=float(getattr(FLAGS, "env_country_weight", 0.2) or 0.0),
                region_weight=float(getattr(FLAGS, "env_region_weight", 0.0) or 0.0),
                city_weight=float(getattr(FLAGS, "env_city_weight", 0.0) or 0.0),
                format_bonus_base=float(getattr(FLAGS, "env_format_bonus_base", 0.05) or 0.0),
                format_bonus_per_field=float(getattr(FLAGS, "env_format_bonus_per_field", 0.0) or 0.0),
                presence_bonus_scale=float(getattr(FLAGS, "env_presence_bonus_scale", 0.0) or 0.0),
                coords_presence_bonus=float(getattr(FLAGS, "env_coords_presence_bonus", 0.0) or 0.0),
            )
            if max_samples and max_samples > 0:
                kwargs["max_samples"] = max_samples
            return create_env(name, tokenizer, **kwargs)
        return create_env(name, tokenizer)

    env = _mk_env(FLAGS.env_name)
    env_test = _mk_env(FLAGS.test_env_name) if FLAGS.test_env_name else None

    sampler = VLMSampler(model, train_state.params)
    image_pad_id = _resolve_image_pad_id(tokenizer, FLAGS.model_dir)

    # Vision pixel bounds
    vlm_min_pixels = FLAGS.vlm_min_pixels if hasattr(FLAGS, "vlm_min_pixels") else -1
    vlm_max_pixels = FLAGS.vlm_max_pixels if hasattr(FLAGS, "vlm_max_pixels") else -1
    vlm_min_pixels = None if int(vlm_min_pixels) <= 0 else int(vlm_min_pixels)
    vlm_max_pixels = None if int(vlm_max_pixels) <= 0 else int(vlm_max_pixels)

    # Sampling knobs
    top_k = int(FLAGS.top_k) if int(getattr(FLAGS, "top_k", -1) or -1) > 0 else None
    top_p = float(FLAGS.top_p)
    top_p = top_p if (0.0 < top_p < 1.0) else None
    temperature = float(getattr(FLAGS, "temperature", 1.0) or 1.0)

    # Sequence length cap
    max_seq_len = int(getattr(FLAGS, "max_sequence_length", 0) or 0)
    if max_seq_len <= 0:
        max_seq_len = None

    rollout_batch_size = jax.local_device_count() * int(FLAGS.inference_batch_per_device)
    assert rollout_batch_size % int(FLAGS.group_size) == 0, "rollout_batch_size must be divisible by group_size"

    rng_global = jax.random.PRNGKey(jax.process_index())
    env_task_idx = 0
    start_step = int(jax.device_get(train_state.step))
    total_steps = int(FLAGS.total_steps)

    # Optional resume (params and opt state)
    resume_dir = str(getattr(FLAGS, "resume_dir", "") or "")
    resume_step_flag = int(getattr(FLAGS, "resume_step", 0) or 0)
    if resume_dir:
        try:
            cp = Checkpoint(resume_dir if resume_dir.endswith(".pkl") else f"{resume_dir}/params.pkl", parallel=False)
            train_state = train_state.replace(params=cp.load())
            if jax.process_index() == 0:
                print(f"Resumed params from {resume_dir}")
            if resume_step_flag > 0:
                train_state = train_state.replace(step=resume_step_flag)
        except Exception as e:
            if jax.process_index() == 0:
                print(f"Resume failed: {e}")
    elif resume_step_flag > 0:
        train_state = train_state.replace(step=resume_step_flag)

    if start_step >= total_steps:
        if jax.process_index() == 0:
            print(f"No steps to run: start_step={start_step} >= total_steps={total_steps}")
        return

    @partial(jax.jit, donate_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
    def _ppo_update(train_state: TrainState,
                    image_pad_id: int,
                    tokens: jnp.ndarray,
                    mask_origin: jnp.ndarray,
                    advantages: jnp.ndarray,
                    old_logprobs: jnp.ndarray,
                    vision_embeds: Union[jnp.ndarray, VisionEmbeddings],
                    token_mask: jnp.ndarray,
                    cos: jnp.ndarray,
                    sin: jnp.ndarray,
                    kl_coef_val: float,
                    kl_target_val: float,
                    kl_scale_min_val: float,
                    kl_scale_updates_flag: int,
                    entropy_coef_val: float,
                    entropy_target_adv_std_val: float,
                    entropy_scale_min_val: float,
                    entropy_scale_by_adv_flag: int):
        text_input = tokens[:, :-1]
        text_target = tokens[:, 1:]

        def loss_fn(grad_params):
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
                    method=train_state.model_def.forward_vlm,
                )
            if getattr(FLAGS, 'gradient_checkpointing', False):
                logits, _ = jax.checkpoint(call_with_params, prevent_cse=False)(grad_params)
            else:
                logits, _ = call_with_params(grad_params)
            # Memory-efficient token logprobs and (optionally) entropy
            lse = jax.scipy.special.logsumexp(logits, axis=-1)  # [B, T-1]
            token_logits = jnp.take_along_axis(logits, text_target[..., None], axis=-1)[..., 0]
            token_logprobs = token_logits - lse
            # If entropy regularization is off, skip computing softmax(probs)
            if float(getattr(FLAGS, 'entropy_coef', 0.0) or 0.0) > 0.0:
                probs = jax.nn.softmax(logits, axis=-1)
                entropy = lse - jnp.sum(probs * logits, axis=-1)
            else:
                entropy = jnp.zeros_like(token_logprobs)

            # PPO ratio & clipping
            logratio = token_logprobs - old_logprobs
            ratio = jnp.exp(logratio)
            if int(getattr(FLAGS, 'do_ppo_all_clip', 0) or 0) == 1:
                pg = -advantages[:, None] * jnp.clip(ratio, 1 - float(FLAGS.clip_epsilon), 1 + float(FLAGS.clip_epsilon))
            else:
                pg1 = -advantages[:, None] * ratio
                pg2 = -advantages[:, None] * jnp.clip(ratio, 1 - float(FLAGS.clip_epsilon), 1 + float(FLAGS.clip_epsilon))
                pg = jnp.maximum(pg1, pg2)

            # Optional masking based on inference/recompute ratio
            mask = mask_origin
            # The caller can precompute and fold-in extra masks if desired.

            denom = jnp.sum(mask) + 1e-8
            loss_pg = jnp.sum(pg * mask) / denom
            # Entropy coefficient with advantage-aware scaling (to avoid random drift when no signal)
            ent_coef = jnp.asarray(entropy_coef_val, dtype=jnp.float32)
            # JAX-safe gating for entropy scaling by advantage std
            _ent_flag = jnp.asarray(entropy_scale_by_adv_flag, dtype=jnp.int32)
            _adv_std_local = jnp.std(advantages.astype(jnp.float32))
            _target_adv = jnp.asarray(entropy_target_adv_std_val, dtype=jnp.float32)
            _emin = jnp.asarray(entropy_scale_min_val, dtype=jnp.float32)
            _scale_if_enabled = jnp.where(
                _target_adv > 0,
                jnp.clip(_adv_std_local / (_target_adv + 1e-8), _emin, 1.0),
                1.0,
            )
            ent_coef = ent_coef * jnp.where(_ent_flag == 1, _scale_if_enabled, 1.0)
            loss_ent = -ent_coef * jnp.sum(entropy * mask) / denom

            # Behavior KL regularization (old||new) = E_old[log p_old - log p_new]
            kl_term = jnp.sum((old_logprobs - token_logprobs) * mask) / denom
            loss_kl = jnp.asarray(kl_coef_val, dtype=jnp.float32) * kl_term

            loss = loss_pg + loss_ent + loss_kl

            # Diagnostics
            approx_kl = jnp.sum(((ratio - 1.0) - logratio) * mask) / denom
            clip_fraction = jnp.sum((jnp.abs(ratio - 1.0) > float(FLAGS.clip_epsilon)) * mask) / denom
            metrics = {
                'loss': loss,
                'loss_pg': loss_pg,
                'loss_ent': loss_ent,
                'loss_kl': loss_kl,
                'entropy': jnp.sum(entropy * mask) / denom,
                'approx_kl': approx_kl,
                'clip_fraction': clip_fraction,
                'token_logprob_mean': jnp.sum(token_logprobs * mask) / denom,
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        # Trust-region gradient scaling when KL exceeds target
        approx_kl = metrics.get('approx_kl')
        if approx_kl is None:
            approx_kl = jnp.asarray(0.0, dtype=jnp.float32)
        # JAX-safe gating for KL-based gradient scaling
        _kl_flag = jnp.asarray(kl_scale_updates_flag, dtype=jnp.int32)
        _target = jnp.asarray(kl_target_val, dtype=jnp.float32)
        _smin = jnp.asarray(kl_scale_min_val, dtype=jnp.float32)
        _safe_scale = jnp.where(approx_kl > _target,
                                jnp.clip(_target / (approx_kl + 1e-8), _smin, 1.0),
                                1.0)
        _scale_mult = jnp.where(_kl_flag == 1, _safe_scale, 1.0)
        grads = jax.tree_util.tree_map(lambda g: g * _scale_mult, grads)
        updates, opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)
        new_state = train_state.replace(params=new_params, opt_state=opt_state, step=train_state.step + 1)
        metrics = {**metrics, 'grad_norm': optax.global_norm(grads), 'update_norm': optax.global_norm(updates)}
        return new_state, metrics

    # Initialize dynamic KL coefficient
    kl_coef_dynamic = float(getattr(FLAGS, "kl_coef", 0.0) or 0.0)
    kl_target = float(getattr(FLAGS, "kl_target", 0.02) or 0.02)
    kl_adapt_rate = float(getattr(FLAGS, "kl_adapt_rate", 1.5) or 1.5)
    kl_coef_min = float(getattr(FLAGS, "kl_coef_min", 1e-5) or 1e-5)
    kl_coef_max = float(getattr(FLAGS, "kl_coef_max", 0.5) or 0.5)

    for step_idx in range(start_step, total_steps):
        if step_idx == start_step and jax.process_index() == 0:
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")

        # Choose params for sampling (EMA can stabilize rollouts)
        if (
            bool(int(getattr(FLAGS, "sample_with_ema", 1) or 1) == 1)
            and bool(int(getattr(FLAGS, "use_ema", 0) or 0) == 1)
            and (train_state.params_ema is not None)
        ):
            sampler.params = train_state.params_ema
        else:
            sampler.params = train_state.params

        # Fill grouped on-policy buffer
        buffer_groups_tokens: List[jnp.ndarray] = []  # each: [group_size, T]
        buffer_groups_infer_logp: List[jnp.ndarray] = []  # [group_size, T]
        buffer_groups_prompt_lens: List[List[int]] = []  # list of group prompt lens
        buffer_groups_action_lens: List[List[int]] = []
        buffer_groups_vision: List[List[Union[jnp.ndarray, VisionEmbeddings]]] = []  # list of group embeds (one per sample)
        buffer_groups_grid: List[List[jnp.ndarray]] = []
        buffer_groups_text_preview: List[str] = []
        env_infos_history: Dict[str, List[float]] = {}
        env_infos_history['return'] = []

        groups_collected = 0
        rollout_iters = 0
        t_rollout_start = time.time()
        env_num_tasks = env.num_tasks if env.num_tasks != -1 else 1_000_000

        while groups_collected < int(FLAGS.groups_per_batch):
            rollout_iters += 1
            # Number of prompts to sample this iteration
            num_prompts = (rollout_batch_size // int(FLAGS.group_size))
            states: List = []
            observations: List = []
            for _ in range(num_prompts):
                idx = min(env_task_idx + jax.process_index(), env_num_tasks - 1)
                env_state, obs = env.reset(idx)
                env_task_idx = (env_task_idx + jax.process_count()) % env_num_tasks
                states.append(env_state)
                observations.append(obs)

            # For each prompt, sample group_size actions independently
            for env_state, obs in zip(states, observations, strict=True):
                prompt_tokens, embeds, grid = _prepare_prompt_and_embeds(
                    model,
                    train_state,
                    tokenizer,
                    obs,
                    vlm_min_pixels,
                    vlm_max_pixels,
                )
                prompt_len = int(prompt_tokens.shape[1])

                group_tokens_list: List[jnp.ndarray] = []
                group_logp_list: List[jnp.ndarray] = []
                group_prompt_lens: List[int] = []
                group_action_lens: List[int] = []
                group_vision_embeds: List[Union[jnp.ndarray, VisionEmbeddings]] = []
                group_grids: List[jnp.ndarray] = []
                group_texts: List[str] = []
                group_generated_tokens: List[jnp.ndarray] = []

                group_size = int(FLAGS.group_size)
                prompt_tokens_batched = jnp.repeat(prompt_tokens, group_size, axis=0)
                if isinstance(embeds, VisionEmbeddings):
                    vision_batched_group = embeds.with_batch_dim(group_size)
                else:
                    vision_arr = jnp.asarray(embeds)
                    if vision_arr.ndim == 2:
                        vision_arr = jnp.repeat(vision_arr[None, ...], group_size, axis=0)
                    elif vision_arr.ndim == 3:
                        if vision_arr.shape[0] == 1 and group_size > 1:
                            vision_arr = jnp.tile(vision_arr, (group_size, 1, 1))
                        elif vision_arr.shape[0] != group_size:
                            raise ValueError("vision embeddings batch dimension mismatch")
                    else:
                        raise ValueError("Unsupported vision embeddings shape for batching")
                    vision_batched_group = vision_arr

                grid_arr = jnp.asarray(grid)
                if grid_arr.ndim == 2:
                    grid_batched_group = jnp.tile(grid_arr[None, ...], (group_size, 1, 1))
                elif grid_arr.ndim == 3:
                    if grid_arr.shape[0] == 1 and group_size > 1:
                        grid_batched_group = jnp.tile(grid_arr, (group_size, 1, 1))
                    else:
                        grid_batched_group = grid_arr
                else:
                    raise ValueError("grid must have shape [num_images, 3] or [batch, num_images, 3]")

                rng_global, group_key = jax.random.split(rng_global)
                generated_batch, sample_logprobs_batch = sampler.sample_vlm(
                    prompt_tokens_batched,
                    vision_batched_group,
                    grid_batched_group,
                    image_pad_id=image_pad_id,
                    max_new_tokens=int(FLAGS.num_generation_tokens),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    rng=group_key,
                    return_logprobs=True,
                )

                prompt_arr_base = prompt_tokens[0]

                for member_idx in range(group_size):
                    gen_tokens = generated_batch[member_idx]
                    gen_logprobs = sample_logprobs_batch[member_idx]
                    prompt_arr = prompt_arr_base
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

                    group_tokens_list.append(full_tokens)
                    group_logp_list.append(full_logprobs)
                    group_prompt_lens.append(prompt_len_effective)
                    group_action_lens.append(int(gen_tokens.shape[0]))
                    group_vision_embeds.append(embeds)
                    group_grids.append(grid)
                    group_generated_tokens.append(gen_tokens)
                    if not group_texts:
                        decoded = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
                        group_texts.append(decoded)

                # Pad within group to uniform length
                group_tokens = _pad_sequences(group_tokens_list, pad_id, max_seq_len)
                group_infer_logp = _pad_float_sequences(group_logp_list, max_len=max_seq_len)
                if max_seq_len is not None:
                    target_len = int(max_seq_len)
                    if group_tokens.shape[1] < target_len:
                        pad_width = int(target_len - group_tokens.shape[1])
                        group_tokens = jnp.pad(
                            group_tokens,
                            ((0, 0), (0, pad_width)),
                            mode="constant",
                            constant_values=pad_id,
                        )
                        group_infer_logp = jnp.pad(
                            group_infer_logp,
                            ((0, 0), (0, pad_width)),
                            mode="constant",
                            constant_values=0.0,
                        )

                # Env step for the group
                actions = [t.tolist() for t in group_generated_tokens]
                states_rep = [env_state for _ in range(int(FLAGS.group_size))]
                _, _, returns_local, _, env_infos = env.step_list(states_rep, actions)
                returns_np = np.asarray(returns_local, dtype=np.float32)
                returns_global = host_gather(shard_data_fn(returns_np))

                # Track env infos
                for k, v in (env_infos or {}).items():
                    if v is None:
                        continue
                    # Only aggregate numeric env infos; skip strings/objects
                    try:
                        arr = np.asarray(v)
                    except (TypeError, ValueError):
                        continue
                    if arr.size == 0 or arr.dtype.kind not in {"b", "i", "u", "f"}:
                        continue
                    arr = arr.astype(np.float32, copy=False).reshape(-1)
                    # Some env metrics are scalars (shape [1]); sharding requires
                    # the first dim to be divisible by local_device_count(). In that
                    # case, skip sharding and aggregate locally (single-host safe).
                    try:
                        if arr.shape[0] % max(1, jax.local_device_count()) == 0 and arr.shape[0] > 0:
                            v_arr = host_gather(shard_data_fn(arr))
                        else:
                            v_arr = arr
                    except Exception:
                        # Fallback to local aggregation if sharding is not possible
                        v_arr = arr
                    env_infos_history.setdefault(k, []).extend(np.asarray(v_arr, dtype=np.float32).reshape(-1).tolist())
                env_infos_history['return'].extend(returns_global.tolist())

                # Append group data
                buffer_groups_tokens.append(group_tokens)
                buffer_groups_infer_logp.append(group_infer_logp)
                buffer_groups_prompt_lens.append(group_prompt_lens)
                buffer_groups_action_lens.append(group_action_lens)
                buffer_groups_vision.append(group_vision_embeds)
                buffer_groups_grid.append(group_grids)
                buffer_groups_text_preview.append(group_texts[0] if group_texts else "")
                groups_collected += 1
                if groups_collected >= int(FLAGS.groups_per_batch):
                    break

        if buffer_groups_tokens:
            if max_seq_len is not None:
                target_len = int(max_seq_len)
            else:
                target_len = max(int(arr.shape[1]) for arr in buffer_groups_tokens)
            for idx in range(len(buffer_groups_tokens)):
                arr_tokens = buffer_groups_tokens[idx]
                arr_logp = buffer_groups_infer_logp[idx]
                if arr_tokens.shape[1] < target_len:
                    pad_width = int(target_len - arr_tokens.shape[1])
                    buffer_groups_tokens[idx] = jnp.pad(
                        arr_tokens,
                        ((0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=pad_id,
                    )
                    buffer_groups_infer_logp[idx] = jnp.pad(
                        arr_logp,
                        ((0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=0.0,
                    )
                elif arr_tokens.shape[1] > target_len:
                    buffer_groups_tokens[idx] = arr_tokens[:, :target_len]
                    buffer_groups_infer_logp[idx] = arr_logp[:, :target_len]

        # Flatten groups -> per-sample tensors
        tokens_all = jnp.concatenate(buffer_groups_tokens, axis=0)  # [N, T]
        inference_logprobs_all = jnp.concatenate(buffer_groups_infer_logp, axis=0)  # [N, T]
        prompt_lens_all = np.asarray([pl for group in buffer_groups_prompt_lens for pl in group], dtype=np.int32)
        action_lens_all = np.asarray([al for group in buffer_groups_action_lens for al in group], dtype=np.int32)

        # Flatten embeds and grids
        vision_list_flat: List[Union[jnp.ndarray, VisionEmbeddings]] = [v for group in buffer_groups_vision for v in group]
        grid_list_flat: List[jnp.ndarray] = [g for group in buffer_groups_grid for g in group]
        vision_batched = _pad_vision(vision_list_flat)
        grid_batched = _pad_grid([np.asarray(g) for g in grid_list_flat])

        # Mask over targets (exclude prompt positions)
        mask_targets = (np.asarray(_build_masks(prompt_lens_all, action_lens_all, int(tokens_all.shape[1]))) > 0).astype(np.int32)

        # Advantages (per-group returns -> normalize)
        returns_arr = np.asarray(env_infos_history['return'], dtype=np.float32)
        num_groups = len(buffer_groups_tokens)
        group_sz = int(FLAGS.group_size)
        assert returns_arr.shape[0] == num_groups * group_sz, "returns size mismatch"
        returns_grouped = returns_arr.reshape(num_groups, group_sz)
        advantages_grouped = returns_grouped.copy()
        if int(getattr(FLAGS, 'do_group_normalization', 1) or 1) == 1:
            mean = advantages_grouped.mean(axis=1, keepdims=True)
            std = advantages_grouped.std(axis=1, keepdims=True) + 1e-8
            advantages_grouped = (advantages_grouped - mean) / std
        if int(getattr(FLAGS, 'do_global_normalization', 0) or 0) == 1:
            gmean = advantages_grouped.mean()
            gstd = advantages_grouped.std() + 1e-8
            advantages_grouped = (advantages_grouped - gmean) / gstd
        if int(getattr(FLAGS, 'do_clip_advantages', 0) or 0) == 1:
            advantages_grouped = np.clip(advantages_grouped, a_min=0.0, a_max=None)
        advantages_all = jnp.asarray(advantages_grouped.reshape(-1), dtype=jnp.float32)

        # PPO minibatch sharding helper
        def ppo_shard(x: jnp.ndarray) -> jnp.ndarray:
            ppo_mb = int(FLAGS.ppo_minibatch)
            chunks = x.shape[0] // ppo_mb
            x = x[: ppo_mb * chunks]
            x = x.reshape(ppo_mb, chunks, *x.shape[1:])
            host_id = jax.process_index()
            host_count = jax.process_count()
            host_slice = chunks // host_count
            x = x[:, host_id * host_slice : (host_id + 1) * host_slice]
            return shard_data_fn(x)

        # Build token mask for model (on inputs)
        text_input = tokens_all[:, :-1]
        token_mask_inputs = (text_input != pad_id).astype(jnp.int32)

        # Minibatch shard tensors
        tokens_mb = ppo_shard(tokens_all)
        old_logp_mb_full = ppo_shard(inference_logprobs_all)
        mask_mb = ppo_shard(jnp.asarray(mask_targets))
        adv_mb = ppo_shard(advantages_all)
        # For vision/grid, avoid pre-sharding the large arrays; gather per-minibatch.
        token_mask_mb = ppo_shard(token_mask_inputs)

        num_mb = tokens_mb.shape[1]
        if num_mb == 0:
            if jax.process_index() == 0:
                print("[WARN] Skipping PPO update: minibatch count is 0 (ppo_minibatch too large?)")
            continue

        # Recompute logprobs if requested
        use_infer_lp = int(getattr(FLAGS, 'do_inference_logprobs', 0) or 0) == 1
        if not use_infer_lp:
            recalc_lp_list = []
            for j_idx in range(num_mb):
                t_mb = tokens_mb[:, j_idx]
                tm_mb = token_mask_mb[:, j_idx]
                m_mb = mask_mb[:, j_idx]
                # Build per-minibatch indices into the flattened [N, ...] tensors
                ppo_mb = int(FLAGS.ppo_minibatch)
                N = int(tokens_all.shape[0])
                chunks_all = max(1, N // max(1, ppo_mb))
                host_id = jax.process_index()
                host_count = jax.process_count()
                host_slice = max(1, chunks_all // max(1, host_count))
                global_col = host_id * host_slice + int(j_idx)
                indices = jnp.asarray([global_col + k * chunks_all for k in range(ppo_mb)], dtype=jnp.int32)
                # Gather vision/grid for this minibatch
                if isinstance(vision_batched, VisionEmbeddings):
                    v_tokens = jnp.take(vision_batched.tokens, indices, axis=0)
                    v_deep = tuple(jnp.take(layer, indices, axis=0) for layer in vision_batched.deepstack)
                    v_mb = VisionEmbeddings(tokens=v_tokens, deepstack=v_deep)
                else:
                    v_mb = jnp.take(vision_batched, indices, axis=0)
                g_mb = jnp.take(grid_batched, indices, axis=0)
                # Stop VLM pack from becoming a tracer inside checkpoint's Jaxpr
                if isinstance(v_mb, VisionEmbeddings):
                    v_mb_eval = VisionEmbeddings(
                        tokens=jax.lax.convert_element_type(jax.lax.stop_gradient(v_mb.tokens), jnp.bfloat16),
                        deepstack=tuple(jax.lax.convert_element_type(jax.lax.stop_gradient(level), jnp.bfloat16) for level in v_mb.deepstack),
                    )
                else:
                    v_mb_eval = jax.lax.convert_element_type(jax.lax.stop_gradient(v_mb), jnp.bfloat16)
                # Memory saver: compute logprobs only over the minimal target window
                # Determine window [start_idx, end_idx] over targets (length T-1)
                try:
                    tgt_mask = (m_mb > 0)
                    any_on = jnp.any(tgt_mask, axis=0)
                    has_any = bool(jax.device_get(jnp.any(any_on)))
                except Exception:
                    has_any = False

                T = int(t_mb.shape[1])
                if has_any:
                    start_idx = int(jax.device_get(jnp.argmax(any_on.astype(jnp.int32))))
                    last_from_end = int(jax.device_get(jnp.argmax(jnp.flip(any_on, axis=0).astype(jnp.int32))))
                    end_idx = (tgt_mask.shape[1] - 1) - last_from_end

                    t_start = max(0, start_idx)
                    t_end = min(T, end_idx + 2)
                    t_slice = t_mb[:, t_start:t_end]
                    tm_slice = tm_mb[:, t_start : t_end - 1]
                cos_mb, sin_mb = compute_rope_indices_chunked(model, t_slice[:, :-1], g_mb, tm_slice)
                try:
                    cos_mb = jax.lax.convert_element_type(cos_mb, jnp.bfloat16)
                    sin_mb = jax.lax.convert_element_type(sin_mb, jnp.bfloat16)
                except Exception:
                    pass
                    lp_slice = _compute_token_logprobs(
                        train_state,
                        image_pad_id,
                        t_slice,
                        tm_slice,
                        v_mb_eval,
                        cos_mb,
                        sin_mb,
                    )
                    # Scatter into full-length buffer
                    lp_full = jnp.zeros((t_mb.shape[0], T - 1), dtype=jnp.float32)
                    lp_full = lp_full.at[:, start_idx : end_idx + 1].set(lp_slice)
                    recalc_lp_list.append(lp_full)
                else:
                    # No targets active; append zeros to avoid model call
                    recalc_lp_list.append(jnp.zeros((t_mb.shape[0], T - 1), dtype=jnp.float32))
            recalc_logprobs_mb = jnp.stack(recalc_lp_list, axis=1)
        else:
            recalc_logprobs_mb = old_logp_mb_full[:, :, 1:]

        # Training loop over minibatches
        update_time_start = time.time()
        step_metrics_accum: Dict[str, float] = {}
        for j_idx in range(num_mb):
            t_mb = tokens_mb[:, j_idx]
            tm_mb = token_mask_mb[:, j_idx]
            m_mb = mask_mb[:, j_idx]
            adv_local = adv_mb[:, j_idx]
            oldlp_full = old_logp_mb_full[:, j_idx]
            oldlp_targets = oldlp_full[:, 1:] if use_infer_lp else recalc_logprobs_mb[:, j_idx]

            # Memory saver: slice sequence to the smallest window that covers
            # all target tokens in this minibatch. This can drastically reduce
            # sequence length during PPO updates and avoid OOM.
            try:
                tgt_mask = (m_mb > 0)
                any_on = jnp.any(tgt_mask, axis=0)  # [T-1]
                has_any = bool(jax.device_get(jnp.any(any_on)))
                if has_any:
                    # Find first and last active target positions
                    start_idx = int(jax.device_get(jnp.argmax(any_on.astype(jnp.int32))))
                    # last true index: (T-1) - argmax(flip)
                    last_from_end = int(jax.device_get(jnp.argmax(jnp.flip(any_on, axis=0).astype(jnp.int32))))
                    end_idx = (tgt_mask.shape[1] - 1) - last_from_end
                    # Map to token slice [start : end+2] so that text_input spans [start..end+1]
                    t_start = max(0, start_idx)
                    t_end = min(int(t_mb.shape[1]), end_idx + 2)
                    if t_end > t_start:
                        # Slice tokens and masks
                        t_mb = t_mb[:, t_start:t_end]
                        tm_mb = tm_mb[:, t_start : t_end - 1]
                        m_mb = m_mb[:, start_idx : end_idx + 1]
                        oldlp_targets = oldlp_targets[:, start_idx : end_idx + 1]
                # else: keep full sequence (no action tokens in this MB)
            except Exception:
                # If anything goes wrong, fall back to full tensors
                pass

            # Optional extra masks
            if int(getattr(FLAGS, 'do_mask_inference_ratio', 0) or 0) == 1 and not use_infer_lp:
                ratio_recompute_infer = jnp.exp(oldlp_full[:, 1:] - recalc_logprobs_mb[:, j_idx])
                m_mb = m_mb * (jnp.abs(ratio_recompute_infer) - 1.0 < 1.0)
            if int(getattr(FLAGS, 'do_mask_importance_ratio', 0) or 0) == 1:
                # Importance ratio based on chosen oldlp_targets vs future token logprobs will be computed in update; here we skip
                pass

            # Build VisionEmbeddings batch for this minibatch via gather (memory efficient)
            ppo_mb = int(FLAGS.ppo_minibatch)
            N = int(tokens_all.shape[0])
            chunks_all = max(1, N // max(1, ppo_mb))
            host_id = jax.process_index()
            host_count = jax.process_count()
            host_slice = max(1, chunks_all // max(1, host_count))
            global_col = host_id * host_slice + int(j_idx)
            indices = jnp.asarray([global_col + k * chunks_all for k in range(ppo_mb)], dtype=jnp.int32)
            if isinstance(vision_batched, VisionEmbeddings):
                v_tokens = jnp.take(vision_batched.tokens, indices, axis=0)
                v_deep = tuple(jnp.take(layer, indices, axis=0) for layer in vision_batched.deepstack)
                v_mb = VisionEmbeddings(tokens=v_tokens, deepstack=v_deep)
            else:
                v_mb = jnp.take(vision_batched, indices, axis=0)
            g_mb = jnp.take(grid_batched, indices, axis=0)
            cos_mb, sin_mb = compute_rope_indices_chunked(model, t_mb[:, :-1], g_mb, tm_mb)
            try:
                cos_mb = jax.lax.convert_element_type(cos_mb, jnp.bfloat16)
                sin_mb = jax.lax.convert_element_type(sin_mb, jnp.bfloat16)
            except Exception:
                pass

            train_state, mb_metrics = _ppo_update(
                train_state,
                image_pad_id,
                t_mb,
                m_mb,
                adv_local,
                oldlp_targets,
                # Use stopped-gradient bfloat16 embeddings to reduce memory
                (VisionEmbeddings(
                    tokens=jax.lax.convert_element_type(jax.lax.stop_gradient(v_mb.tokens), jnp.bfloat16),
                    deepstack=tuple(jax.lax.convert_element_type(jax.lax.stop_gradient(level), jnp.bfloat16) for level in v_mb.deepstack),
                ) if isinstance(v_mb, VisionEmbeddings) else jax.lax.convert_element_type(jax.lax.stop_gradient(v_mb), jnp.bfloat16)),
                tm_mb,
                cos_mb,
                sin_mb,
                float(kl_coef_dynamic),
                float(kl_target),
                float(getattr(FLAGS, "kl_scale_min", 0.05) or 0.05),
                int(getattr(FLAGS, "kl_scale_updates", 1) or 1),
                float(getattr(FLAGS, "entropy_coef", 0.0) or 0.0),
                float(getattr(FLAGS, "entropy_target_adv_std", 0.3) or 0.3),
                float(getattr(FLAGS, "entropy_scale_min", 0.0) or 0.0),
                int(getattr(FLAGS, "entropy_scale_by_adv", 1) or 1),
            )

            # Update EMA after applying gradients (outside jit for simplicity)
            if bool(int(getattr(FLAGS, "use_ema", 0) or 0) == 1) and (train_state.params_ema is not None):
                tau = float(getattr(FLAGS, "ema_tau", 0.999) or 0.999)
                train_state = train_state.update_ema(tau)

            # Accumulate metrics
            mb_metrics = jax.tree_util.tree_map(lambda x: float(jax.device_get(x)), mb_metrics)
            for k, v in mb_metrics.items():
                step_metrics_accum[k] = step_metrics_accum.get(k, 0.0) + v

        # Average metrics across minibatches
        for k in list(step_metrics_accum.keys()):
            step_metrics_accum[k] /= max(1, num_mb)

        # Logging
        info = {
            'global_step': int(step_idx),
            'rollouts_per_step': float(len(env_infos_history['return'])),
            'times/total_time_rollouts': float(time.time() - t_rollout_start),
            'times/total_time_update': float(time.time() - update_time_start),
            'reward/mean': float(np.mean(env_infos_history['return'])) if env_infos_history['return'] else 0.0,
            'reward/std': float(np.std(env_infos_history['return'])) if env_infos_history['return'] else 0.0,
            'advantage/std': float(np.std(np.asarray(advantages_all))) if advantages_all.size else 0.0,
            'train/kl_coef': float(kl_coef_dynamic),
        }
        info.update({f"train/{k}": float(v) for k, v in step_metrics_accum.items()})
        env_metrics_summary = {}
        for k, v in env_infos_history.items():
            if k == 'return':
                continue
            if not v:
                continue
            try:
                mean_val = float(np.mean(v))
            except Exception:
                continue
            info[f'env/{k}'] = mean_val
            env_metrics_summary[k] = mean_val

        if jax.process_index() == 0:
            # Compact, single-line console log that adapts to terminal width
            term_cols = 120
            try:
                term_cols = int(shutil.get_terminal_size(fallback=(120, 24)).columns)
            except Exception:
                pass

            preview_text = buffer_groups_text_preview[0] if buffer_groups_text_preview else ""
            if preview_text:
                preview_text = preview_text.replace("\n", " ").replace("\t", " ")

            sample_reward = float(returns_arr[0]) if returns_arr.size else float("nan")
            loss_val = float(info.get('train/loss', float('nan')))
            kl_val = float(info.get('train/approx_kl', float('nan')))
            clip_frac = float(info.get('train/clip_fraction', float('nan')))
            ent_val = float(info.get('train/entropy', float('nan')))
            tok_lp = float(info.get('train/token_logprob_mean', float('nan')))
            adv_std = float(info.get('advantage/std', float('nan')))
            rew_mean = float(info.get('reward/mean', float('nan')))
            rew_std = float(info.get('reward/std', float('nan')))
            t_roll = float(info.get('times/total_time_rollouts', float('nan')))
            t_upd = float(info.get('times/total_time_update', float('nan')))

            # Throughput: tokens/sec during rollout (approx)
            try:
                total_tokens = int(np.sum(action_lens_all))
                tps = (total_tokens / t_roll) if t_roll > 0 else float('nan')
            except Exception:
                tps = float('nan')

            # Compact env metrics (selected keys if present)
            kc = env_metrics_summary.get('k_coords')
            kci = env_metrics_summary.get('k_city')
            kr = env_metrics_summary.get('k_region')
            kco = env_metrics_summary.get('k_country')
            tol = env_metrics_summary.get('within_tolerance')
            env_compact_parts = []
            if kc is not None:
                env_compact_parts.append(f"kc={kc:.3f}")
            if kci is not None:
                env_compact_parts.append(f"ci={kci:.3f}")
            if kr is not None:
                env_compact_parts.append(f"rg={kr:.3f}")
            if kco is not None:
                env_compact_parts.append(f"co={kco:.3f}")
            if tol is not None:
                env_compact_parts.append(f"tol={tol:.3f}")
            env_compact = " ".join(env_compact_parts)

            # Build prioritized parts to fit within terminal width
            parts = [
                f"step={step_idx}",
                f"rm={rew_mean:.3f}",
                f"rs={rew_std:.3f}",
                f"rsamp={sample_reward:.3f}",
                f"loss={loss_val:.4f}",
                f"kl={kl_val:.4f}",
                f"cf={clip_frac:.3f}",
                f"ent={ent_val:.3f}",
                f"adv={adv_std:.3f}",
                f"tlp={tok_lp:.3f}",
                f"troll={t_roll:.2f}s",
                f"tupd={t_upd:.2f}s",
                f"tps={tps:.0f}/s" if not np.isnan(tps) else None,
            ]
            parts = [p for p in parts if p is not None]

            line = ""
            first = True
            for p in parts:
                if not p:
                    continue
                candidate = p if first else (" " + p)
                if len(line) + len(candidate) <= term_cols:
                    line += candidate
                    first = False

            # Try to append env metrics if they fit
            if env_compact:
                candidate = f" env[{env_compact}]"
                if len(line) + len(candidate) <= term_cols:
                    line += candidate

            # Try to append truncated response to fill remaining space
            if preview_text:
                # Compute remaining space, accounting for wrapper and quotes
                overhead = len(' resp=""')
                remaining = term_cols - len(line) - overhead
                if remaining > 0:
                    snippet = preview_text[:remaining]
                    # If we truncated, indicate ellipsis
                    if len(preview_text) > remaining and remaining >= 3:
                        snippet = snippet[:-3] + "..."
                    if snippet:
                        line += f" resp=\"{snippet}\""

            print(line)

            wandb.log(info, commit=True)

        # Adaptive KL adjustment (outside jit). Keeps updates in a trust region.
        if int(getattr(FLAGS, "adaptive_kl", 1) or 1) == 1:
            try:
                approx_kl_val = float(info.get('train/approx_kl', 0.0))
            except Exception:
                approx_kl_val = 0.0
            if kl_target > 0:
                # Scale towards target with a gentle multiplicative update
                if approx_kl_val > kl_target * 2.0:
                    kl_coef_dynamic = min(kl_coef_dynamic * kl_adapt_rate, kl_coef_max) if kl_coef_dynamic > 0 else min(1e-4, kl_coef_max)
                elif approx_kl_val < kl_target * 0.5:
                    kl_coef_dynamic = max(kl_coef_dynamic / kl_adapt_rate, kl_coef_min)
                # Hard safety: if KL explodes, increase aggressively once
                if approx_kl_val > max(0.5, 5.0 * kl_target):
                    kl_coef_dynamic = min(kl_coef_dynamic * (kl_adapt_rate ** 2), kl_coef_max)

        # Periodic save
        if step_idx % int(FLAGS.save_interval) == 0 and str(getattr(FLAGS, 'save_dir', '') or '') and jax.process_index() == 0:
            params_gather = host_gather(train_state.params)
            ckpt_dir = f"{FLAGS.save_dir}/step{step_idx}/"
            cp = Checkpoint(f"{ckpt_dir}params.pkl", parallel=False)
            cp.params = params_gather
            cp.save()

        # Periodic test
        if (
            env_test is not None
            and step_idx > 0
            and step_idx % int(FLAGS.test_interval) == 0
            and jax.process_index() == 0
        ):
            # Simple batched sampling for eval (no groups)
            eval_states, eval_obs = [], []
            for _ in range(rollout_batch_size):
                idx = min(env_task_idx + jax.process_index(), env_num_tasks - 1)
                e_state, e_obs = env_test.reset(idx)
                eval_states.append(e_state)
                eval_obs.append(e_obs)

            actions_eval = []
            for e_obs in eval_obs:
                p_tokens, e_embeds, e_grid = _prepare_prompt_and_embeds(
                    model,
                    train_state,
                    tokenizer,
                    e_obs,
                    vlm_min_pixels,
                    vlm_max_pixels,
                )
                rng_global, key = jax.random.split(rng_global)
                generated, _ = sampler.sample_vlm(
                    p_tokens,
                    e_embeds,
                    e_grid,
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
                actions_eval.append(generated[0].tolist())

            _, _, ret_local, _, _ = env_test.step_list(eval_states, actions_eval)
            ret = host_gather(shard_data_fn(np.asarray(ret_local, dtype=np.float32)))
            wandb.log({"test/return": float(np.mean(ret)), "global_step": int(step_idx)}, commit=False)


if __name__ == "__main__":
    app.run(main)
