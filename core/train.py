"""Minimal PPO training CLI using the unified Qwen3-VL abstractions.

This entry point wires together:
- Environment creation via `envs.base.create_env`
- Sampling through `core.sampling`
- Policy scoring utilities from `core.policy`
- PPO collect/update helpers in `core.ppo`

It supports the bundled vision environments (OSV5M or vision_caption) and
expects a Qwen3-VL checkpoint produced by `hf_to_jax`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

# This was annoying and should probably not be off by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Ensure macOS Metal-friendly env vars are set before importing jax.
try:
    from vlmrl.utils.platform import apply_macos_env, apply_macos_train_overrides, is_macos

    apply_macos_env()
except Exception:
    # If platform helper fails for any reason, continue with defaults.
    def is_macos() -> bool:  # type: ignore
        return False

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import wandb
from absl import app, flags
from transformers import AutoTokenizer

from vlmrl.core.kl import AdaptiveKL
from vlmrl.core.ppo import TrainerConfig, collect, update
from vlmrl.envs.base import create_env
from vlmrl.models.qwen3vl.model import Qwen3VLModel, create_model_from_ckpt
from vlmrl.utils.checkpoint import Checkpoint
from vlmrl.utils.configs import define_flag_dict
from vlmrl.utils.train_state import TrainState
from vlmrl.utils.wandb import setup_wandb, define_env_metrics, summarize_env_metrics
from vlmrl.core.trace import compute_update_heat


config = ml_collections.ConfigDict({
    # Logging / bookkeeping
    "wandb_project": "geospot",
    "wandb_name": "ppo-geospot-qwen3vl_4b",
    "wandb_group": "Default",
    "wandb_entity": "",
    "wandb_mode": "online",  # online or offline
    "model_dir": "checkpoints/qwen3vl_4b",
    "save_dir": "runs/ppo-geospot-qwen3vl_4b",
    "save_interval": 20,
    "resume_path": "",
    "seed": 0,

    # Environment
    "env_name": "geospot",
    "env_split": "test",
    "env_max_samples": -1,
    "env_coord_tolerance_km": 25.0,
    "env_use_geo_shaping": 1,
    "env_geo_decay_km": 50.0,
    "env_country_weight": 0.2,
    "env_region_weight": 0.3,
    "env_city_weight": 0.5,

    # Training duration
    "total_steps": 1000,
    "batch_size": 16,
    "log_interval": 1,

    # Sampling / sequence knobs
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 1024,
    "max_new_tokens": 64,
    "max_sequence_length": 2048,
    "vlm_min_pixels": -1,
    "vlm_max_pixels": -1,

    # PPO hyperparameters
    "clip_epsilon": 0.2,
    "entropy_coef": 0.0,
    "kl_coef": 0.0,
    "adaptive_kl": 1,
    "kl_target": 0.02,
    "kl_adapt_rate": 1.5,
    "kl_coef_min": 1e-5,
    "kl_coef_max": 0.5,
    "ppo_minibatch": 64,
    "ppo_epochs": 1,
    # Memory helpers
    "grad_checkpoint": 0,

    # Optimizer
    "optimizer": "adamw",
    "learning_rate": 1e-6,
    "weight_decay": 1e-2,
    "max_grad_norm": 1.0,
    "use_ema": 0,
    # Low-memory mode: tunes defaults for constrained accelerators (e.g., Colab TPU).
    # Applies conservative overrides for memory: bf16 params, Adafactor, small PPO minibatch,
    # stricter vision pixel cap, skip DeepStack staging.
    "low_memory": 0,
})

# Apply macOS-specific default overrides (CLI flags still win).
try:
    apply_macos_train_overrides(config)
except Exception:
    pass

define_flag_dict(config)
FLAGS = flags.FLAGS


@dataclass
class RunState:
    train_state: TrainState
    iteration: int
    rng: jax.Array


class RollingEnv:
    """Wraps env.reset so collect() can advance through the dataset."""

    def __init__(self, env):
        self._env = env
        self._cursor = 0
        self.last_infos: Dict[str, Any] = {}

    @property
    def num_tasks(self) -> int:
        return getattr(self._env, "num_tasks", -1)

    def reset(self, idx):
        total = self.num_tasks
        base = self._cursor + idx
        actual = base % total if total and total > 0 else base
        return self._env.reset(actual)

    def step_list(self, states, actions):
        result = self._env.step_list(states, actions)
        self.last_infos = result[4]
        batch = len(states)
        total = self.num_tasks
        self._cursor += batch
        if total and total > 0:
            self._cursor %= total
        return result

    def __getattr__(self, name):
        return getattr(self._env, name)


def _resolve_pad_and_eos(tokenizer, model: Qwen3VLModel) -> tuple[int, Optional[int]]:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(model.spec, "pad_token_id", 0)
    pad = int(pad or 0)

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.spec, "eos_token_id", None)
    return pad, (int(eos) if eos is not None else None)


def _resolve_image_pad_id(tokenizer, ckpt_dir: str) -> int:
    """Resolve the special image placeholder token id."""
    try:
        image_pad = getattr(tokenizer, "image_token_id", None)
        if image_pad is not None and int(image_pad) >= 0:
            return int(image_pad)
        special = getattr(tokenizer, "special_tokens_map", None)
        if isinstance(special, dict):
            image_pad = special.get("image_token_id", None)
            if image_pad is not None and int(image_pad) >= 0:
                return int(image_pad)
    except Exception:
        pass
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        image_pad = cfg.get("image_token_id", 151655)
        return int(image_pad)
    return 151655


def _maybe(val: float | int) -> Optional[int]:
    return int(val) if int(val) > 0 else None


def _format_ground_truth(env_infos: Dict[str, Any]) -> str:
    if not isinstance(env_infos, dict):
        return ""
    ground_truth = env_infos.get("ground_truth", None)
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else None
    if not isinstance(ground_truth, dict):
        return ""
    city = ground_truth.get("city")
    region = ground_truth.get("region")
    country = ground_truth.get("country")
    coords = ground_truth.get("coords")
    lat = lon = None
    if isinstance(coords, (tuple, list)) and len(coords) >= 2:
        lat, lon = coords[0], coords[1]
    parts = []
    if city:
        parts.append(f"City: {city}")
    if region:
        parts.append(f"Region: {region}")
    if country:
        parts.append(f"Country: {country}")
    if lat is not None:
        parts.append(f"Latitude: {lat}")
    if lon is not None:
        parts.append(f"Longitude: {lon}")
    return " ".join(parts)


def _make_optimizer() -> optax.GradientTransformation:
    lr = float(FLAGS.learning_rate)
    weight_decay = float(FLAGS.weight_decay)
    grad_clip = float(FLAGS.max_grad_norm)
    name = str(FLAGS.optimizer).lower()

    if name == "adafactor":
        base_opt = optax.adafactor(
            learning_rate=lr,
            min_dim_size_to_factor=32,
            weight_decay_rate=(weight_decay if weight_decay > 0 else None),
            dtype_momentum=jnp.bfloat16,
        )
    else:
        base_opt = optax.adamw(
            learning_rate=lr,
            weight_decay=weight_decay if weight_decay > 0 else 0.0,
        )

    chain = []
    if grad_clip and grad_clip > 0:
        chain.append(optax.clip_by_global_norm(grad_clip))
    chain.append(base_opt)
    return optax.chain(*chain)


def _cast_params_bf16_except_lm_head(params):
    """Cast parameter pytree to bf16 except keep lm_head/kernel in fp32 for logits stability."""
    try:
        from flax.core import unfreeze, freeze
    except Exception:
        return params

    pf = unfreeze(params)

    def _recurse(obj, path=()):
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = _recurse(v, path + (k,))
            return obj
        try:
            # Arrays (JAX/NumPy)
            if len(path) >= 2 and path[-2] == "lm_head" and path[-1] == "kernel":
                return jnp.asarray(obj, dtype=jnp.float32)
            return jnp.asarray(obj, dtype=jnp.bfloat16)
        except Exception:
            return obj

    pf = _recurse(pf)
    return freeze(pf)


def _build_env(tokenizer):
    name = str(FLAGS.env_name or "").lower()
    max_samples = int(getattr(FLAGS, "env_max_samples", -1) or -1)
    kwargs: Dict[str, Any] = {}
    if name == "osv5m":
        kwargs.update({
            "split": str(getattr(FLAGS, "env_split", "test") or "test"),
            "coord_tolerance_km": float(getattr(FLAGS, "env_coord_tolerance_km", 25.0) or 25.0),
            "country_weight": float(getattr(FLAGS, "env_country_weight", 0.2) or 0.0),
            "region_weight": float(getattr(FLAGS, "env_region_weight", 0.3) or 0.0),
            "city_weight": float(getattr(FLAGS, "env_city_weight", 0.5) or 0.0),
            # Mid-Phase (3–8B or stable) defaults
            "difficulty_schedule": {0: "country", 1500: "region", 3000: "city"},
            "presence_mode": "progressive",
            "coupling_mode": "thresholded",
            "coord_unlock_threshold": 0.5,
            "text_tau_km": 250.0,
            "coord_tau_km": 20.0,
            "coords_presence_bonus": 0.25,
            "use_online_geocoder": True,
            # Penalties
            "allow_negative_reward": False,
            "penalty_missing_field": 0.02,
            "penalty_invalid_coord": 0.03,
            "penalty_wrong_country": 0.03,
            "penalty_wrong_region": 0.02,
            "penalty_wrong_city": 0.01,
            "penalty_hemisphere": 0.03,
            "penalty_text_coord_mismatch": 0.03,
        })
        if max_samples > 0:
            kwargs["max_samples"] = max_samples
    elif name == "geospot":
        kwargs["split"] = str(getattr(FLAGS, "env_split", "test") or "test")
    return RollingEnv(create_env(name, tokenizer, **kwargs))


def _setup_train_state(model: Qwen3VLModel, params, rng: jax.Array) -> TrainState:
    tx = _make_optimizer()
    use_ema = bool(int(getattr(FLAGS, "use_ema", 0) or 0) == 1)
    train_state = TrainState.create_with_params(
        rng=rng,
        model_def=model,
        params=params,
        tx=tx,
        use_ema=use_ema,
    )
    return jax.device_put(train_state)


def _maybe_resume(run_state: RunState, resume_path: str) -> RunState:
    if not resume_path:
        return run_state
    if not os.path.exists(resume_path):
        if jax.process_index() == 0:
            print(f"[train] Resume path not found: {resume_path}")
        return run_state
    try:
        cp = Checkpoint(resume_path, parallel=False)
        data = cp.load_as_dict()
        if "train_state" in data:
            train_state = run_state.train_state.load(data["train_state"])
        else:
            train_state = run_state.train_state.replace(**data)
        iteration = int(np.asarray(jax.device_get(train_state.step))) if hasattr(train_state, "step") else run_state.iteration
        if jax.process_index() == 0:
            print(f"[train] Resumed state from {resume_path}")
        return RunState(train_state=train_state, iteration=iteration, rng=run_state.rng)
    except Exception as exc:
        if jax.process_index() == 0:
            print(f"[train] Failed to resume from {resume_path}: {exc}")
        return run_state


def _maybe_save(train_state: TrainState, save_dir: str, step: int) -> None:
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "train_state.pkl")
    cp = Checkpoint(ckpt_path, parallel=False)
    cp.train_state = train_state
    cp.metadata = {"step": step}
    cp.save()


def main(_):
    FLAGS(sys.argv)

    # Prefer the most precise matmul on METAL to reduce numerical drift.
    try:
        from jax import config as jax_config

        if jax.devices() and jax.devices()[0].platform.lower() == "metal":
            jax_config.update("jax_default_matmul_precision", "highest")
    except Exception:
        pass

    # Apply low-memory overrides early so downstream config picks them up.
    low_mem = bool(int(getattr(FLAGS, "low_memory", 0) or 0) == 1)
    if low_mem:
        # Skip staging DeepStack feature copies in training batches.
        os.environ["VLMRL_SKIP_DEEPSTACK"] = "1"
        # Prefer Adafactor for optimizer state efficiency.
        try:
            FLAGS.optimizer = "adafactor"
        except Exception:
            pass
        # Cap pixels to reduce sequence length (tokens) and activation size.
        try:
            if int(getattr(FLAGS, "vlm_max_pixels", -1) or -1) <= 0 or int(FLAGS.vlm_max_pixels) > 120_000:
                FLAGS.vlm_max_pixels = 120_000
        except Exception:
            pass
        # Keep PPO microbatch small but ensure it uses all local devices for pmap.
        try:
            ndev = max(1, int(jax.local_device_count()))
            current_mb = int(getattr(FLAGS, "ppo_minibatch", 64) or 64)
            # Do not downscale if user set a larger value; only bump up to at least ndev.
            if current_mb < ndev:
                FLAGS.ppo_minibatch = ndev
        except Exception:
            pass
        # Enable gradient checkpointing to reduce activation memory during backward.
        try:
            FLAGS.grad_checkpoint = 1
        except Exception:
            pass
        # Collection batch can also be trimmed on small-memory devices.
        try:
            if int(getattr(FLAGS, "batch_size", 16) or 16) > 8:
                FLAGS.batch_size = 8
        except Exception:
            pass
        # Avoid entropy compute by default; rely on sampling + KL for exploration.
        try:
            FLAGS.entropy_coef = 0.0
        except Exception:
            pass

    if jax.process_index() == 0:
        print(f"[train] Loading model from {FLAGS.model_dir}")
    model, params = create_model_from_ckpt(FLAGS.model_dir)
    # Cast params to bf16 in low-memory mode to halve footprint (keep lm_head/kernel fp32).
    if low_mem:
        try:
            params = _cast_params_bf16_except_lm_head(params)
            if jax.process_index() == 0:
                print("[train] low_memory=1: cast parameters to bfloat16 (lm_head in fp32)")
        except Exception as exc:
            if jax.process_index() == 0:
                print(f"[train] bf16 cast skipped due to: {exc}")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_dir, trust_remote_code=False)
    pad_id, eos_id = _resolve_pad_and_eos(tokenizer, model)
    image_pad_id = _resolve_image_pad_id(tokenizer, FLAGS.model_dir)

    rng = jax.random.PRNGKey(int(FLAGS.seed))
    run_state = RunState(train_state=_setup_train_state(model, params, rng), iteration=0, rng=rng)
    run_state = _maybe_resume(run_state, str(getattr(FLAGS, "resume_path", "") or ""))

    env = _build_env(tokenizer)

    trainer_cfg = TrainerConfig(
        pad_id=pad_id,
        eos_id=eos_id,
        image_pad_id=image_pad_id,
        temperature=float(FLAGS.temperature),
        top_p=(float(FLAGS.top_p) if 0.0 < float(FLAGS.top_p) < 1.0 else None),
        top_k=(int(FLAGS.top_k) if int(FLAGS.top_k) > 0 else None),
        max_new_tokens=int(FLAGS.max_new_tokens),
        max_sequence_length=_maybe(int(getattr(FLAGS, "max_sequence_length", 0) or 0)),
        vlm_min_pixels=_maybe(int(getattr(FLAGS, "vlm_min_pixels", -1) or -1)),
        vlm_max_pixels=_maybe(int(getattr(FLAGS, "vlm_max_pixels", -1) or -1)),
        clip_epsilon=float(FLAGS.clip_epsilon),
        entropy_coef=float(FLAGS.entropy_coef),
        kl_coef=float(FLAGS.kl_coef),
        ppo_minibatch=int(FLAGS.ppo_minibatch),
        num_epochs=int(FLAGS.ppo_epochs),
    )

    kl_ctrl = None
    if int(getattr(FLAGS, "adaptive_kl", 0) or 0) == 1:
        kl_ctrl = AdaptiveKL(
            target=float(getattr(FLAGS, "kl_target", 0.02) or 0.02),
            rate=float(getattr(FLAGS, "kl_adapt_rate", 1.5) or 1.5),
            coef_min=float(getattr(FLAGS, "kl_coef_min", 1e-5) or 1e-5),
            coef_max=float(getattr(FLAGS, "kl_coef_max", 0.5) or 0.5),
        )

    wandb_run = None
    if jax.process_index() == 0:
        try:
            wandb_run = setup_wandb(
                FLAGS.flag_values_dict(),
                project=FLAGS.wandb_project,
                name=FLAGS.wandb_name,
                group=FLAGS.wandb_group,
                entity=(FLAGS.wandb_entity or None) or None,
                offline=str(FLAGS.wandb_mode).lower() == "offline",
            )
            define_env_metrics()
        except Exception as exc:
            print(f"[train] W&B setup failed: {exc}")

    total_steps = int(FLAGS.total_steps)
    batch_size = int(FLAGS.batch_size)
    save_interval = int(getattr(FLAGS, "save_interval", 0) or 0)
    log_interval = max(1, int(getattr(FLAGS, "log_interval", 1) or 1))

    for step_idx in range(run_state.iteration, total_steps):
        run_state.iteration = step_idx
        run_state.rng, key = jax.random.split(run_state.rng)

        rollout, batch = collect(
            env,
            tokenizer,
            model,
            run_state.train_state,
            trainer_cfg,
            batch_size=batch_size,
            rng=key,
        )

        run_state.train_state, metrics = update(
            run_state.train_state,
            trainer_cfg.image_pad_id,
            batch,
            rollout,
            clip_epsilon=trainer_cfg.clip_epsilon,
            entropy_coef=trainer_cfg.entropy_coef,
            kl_coef=trainer_cfg.kl_coef,
            minibatch_size=trainer_cfg.ppo_minibatch,
            num_epochs=trainer_cfg.num_epochs,
            kl_ctrl=kl_ctrl,
            use_checkpoint=bool(int(getattr(FLAGS, "grad_checkpoint", 0) or 0) == 1),
        )

        returns = np.asarray(jax.device_get(rollout.returns))
        reward_mean = float(returns.mean()) if returns.size else 0.0
        reward_std = float(returns.std()) if returns.size else 0.0

        log_step = step_idx + 1
        env_metrics = summarize_env_metrics(getattr(env, "last_infos", {}) or {})
        metrics = {**metrics, **env_metrics}
        metrics.update({
            "reward/mean": reward_mean,
            "reward/std": reward_std,
            "global_step": log_step,
        })

        if rollout.texts:
            metrics["sample/text"] = rollout.texts[0]

        if jax.process_index() == 0 and (log_step % log_interval == 0):
            response_preview = ""
            if rollout.texts:
                response_preview = rollout.texts[0].replace("\n", " ").strip()
            to_print = {
                "step": log_step,
                "reward": f"{reward_mean:.3f}",
                "approx_kl": f"{metrics.get('approx_kl', 0.0):.4f}",
                "loss": f"{metrics.get('loss', 0.0):.4f}",
                "entropy": f"{metrics.get('entropy', 0.0):.3f}",
            }
            to_print["response"] = response_preview
            # gt_preview = _format_ground_truth(getattr(env, "last_infos", {}) or {})
            # if gt_preview:
            #     to_print["ground_truth"] = gt_preview
            print("[train]", to_print)
            if wandb_run is not None:
                wandb.log(metrics, step=log_step)

        if save_interval > 0 and log_step % save_interval == 0 and jax.process_index() == 0:
            _maybe_save(run_state.train_state, FLAGS.save_dir, log_step)

    if jax.process_index() == 0:
        _maybe_save(run_state.train_state, FLAGS.save_dir, total_steps)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    app.run(main)
