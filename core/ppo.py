"""Minimal PPO learner built on the new abstractions.

This module intentionally keeps concerns separate:
- Collection uses the unified sampler (core.sampling.sample) and batching helpers.
- Policy scoring uses core.policy.token_logprobs and core.policy.build_rope.
- Update applies a clipped PPO objective with optional KL and entropy terms.

It coexists with the existing trainers without modifying them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from vlmrl.core.types import SamplingConfig, Rollout, Batch
from vlmrl.core.policy import token_logprobs as policy_token_logprobs, build_rope
from vlmrl.core.rollout import collect_episodes, episodes_to_training_batch
from vlmrl.core.kl import AdaptiveKL
from vlmrl.models.qwen3vl.model import Qwen3VLModel, VisionEmbeddings
from vlmrl.utils.train_state import TrainState


@dataclass
class TrainerConfig:
    pad_id: int
    eos_id: Optional[int]
    image_pad_id: int
    temperature: float = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 1024
    max_new_tokens: int = 64
    max_sequence_length: Optional[int] = 2048
    vlm_min_pixels: Optional[int] = None
    vlm_max_pixels: Optional[int] = None
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    kl_coef: float = 0.0
    ppo_minibatch: int = 64
    num_epochs: int = 1


def _pixels(val: Optional[int]) -> Optional[int]:
    return val if (val is not None and int(val) > 0) else None


def collect(
    env,
    tokenizer,
    model: Qwen3VLModel,
    train_state: TrainState,
    cfg: TrainerConfig,
    *,
    batch_size: int,
    rng: jax.Array,
) -> Tuple[Rollout, Batch]:
    """Collect a batch of on-policy rollouts and return paired Batch for training."""
    sampling_cfg = SamplingConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        eos_id=cfg.eos_id,
        pad_id=cfg.pad_id,
        max_new_tokens=cfg.max_new_tokens,
    )

    episode_batch = collect_episodes(
        env,
        tokenizer,
        model,
        train_state.params,
        sampling_cfg,
        image_pad_id=cfg.image_pad_id,
        batch_size=batch_size,
        rng=rng,
        min_pixels=_pixels(cfg.vlm_min_pixels),
        max_pixels=_pixels(cfg.vlm_max_pixels),
        max_sequence_length=cfg.max_sequence_length,
        return_logprobs=True,
    )

    rollout, batch = episodes_to_training_batch(
        model,
        episode_batch.episodes,
        pad_id=cfg.pad_id,
        max_sequence_length=cfg.max_sequence_length,
    )
    return rollout, batch


def update(
    train_state: TrainState,
    image_pad_id: int,
    batch: Batch,
    rollout: Rollout,
    *,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.0,
    kl_coef: float = 0.0,
    minibatch_size: int = 64,
    num_epochs: int = 1,
    kl_ctrl: Optional[AdaptiveKL] = None,
) -> Tuple[TrainState, dict]:
    """Apply PPO-style updates over minibatches and return new TrainState + metrics."""
    tokens_all = batch.tokens
    mask_targets_all = rollout.mask_targets
    oldlp_all = rollout.old_logprobs[:, 1:]  # align with text targets
    advantages = rollout.returns
    # TODO: Remove this once we have a way to skip normalization for single-sample updates so the lone advantage stays non-zero on single GPU.
    if advantages.shape[0] > 1:
        advantages = advantages - advantages.mean()
        std = advantages.std()
        advantages = jnp.where(std > 1e-8, advantages / std, advantages)

    # One-step jit loss-apply function
    @jax.jit
    def _ppo_step(ts: TrainState,
                  tokens: jnp.ndarray,
                  mask_targets: jnp.ndarray,
                  adv: jnp.ndarray,
                  oldlp: jnp.ndarray,
                  vision: Union[jnp.ndarray, VisionEmbeddings],
                  token_mask: jnp.ndarray,
                  cos: jnp.ndarray,
                  sin: jnp.ndarray,
                  kl_coef_val: float) -> Tuple[TrainState, dict]:
        text_target = tokens[:, 1:]

        def loss_fn(p):
            # Forward with given params
            def call_with_params(params):
                return ts.call_model(
                    tokens[:, :-1],
                    vision,
                    image_pad_id,
                    cos,
                    sin,
                    mask=token_mask,
                    cache=None,
                    params=params,
                    method=ts.model_def.forward_vlm,
                )
            logits, _ = call_with_params(p)
            all_logprobs = jax.nn.log_softmax(logits, axis=-1)
            token_logprobs = jnp.sum(all_logprobs * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)
            entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * all_logprobs, axis=-1)

            logratio = token_logprobs - oldlp
            ratio = jnp.exp(logratio)
            pg1 = -adv[:, None] * ratio
            pg2 = -adv[:, None] * jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            pg = jnp.maximum(pg1, pg2)

            mask = mask_targets
            denom = jnp.sum(mask) + 1e-8
            loss_pg = jnp.sum(pg * mask) / denom
            loss_ent = -jnp.asarray(entropy_coef, dtype=jnp.float32) * jnp.sum(entropy * mask) / denom
            kl_term = jnp.sum((oldlp - token_logprobs) * mask) / denom  # E_old[log p_old - log p_new]
            loss_kl = jnp.asarray(kl_coef_val, dtype=jnp.float32) * kl_term
            loss = loss_pg + loss_ent + loss_kl

            approx_kl = jnp.sum(((ratio - 1.0) - logratio) * mask) / denom
            clip_fraction = jnp.sum((jnp.abs(ratio - 1.0) > clip_epsilon) * mask) / denom
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

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
        updates, opt_state = ts.tx.update(grads, ts.opt_state, ts.params)
        new_params = jax.tree_util.tree_map(lambda p, u: p + u, ts.params, updates)
        new_state = ts.replace(params=new_params, opt_state=opt_state, step=ts.step + 1)
        metrics = {**metrics, 'grad_norm': jnp.asarray(jax.tree_util.tree_reduce(lambda a, b: a + jnp.sum(b*b), grads, 0.0))}
        return new_state, metrics

    # Training loop over epochs and contiguous minibatches
    N = int(tokens_all.shape[0])
    mb = int(minibatch_size)
    chunks = max(1, N // max(1, mb))
    n_use = chunks * mb
    if n_use == 0:
        return train_state, {'skipped': 1.0}

    # Pre-slice tensors common to all minibatches
    tokens_all = tokens_all[:n_use]
    mask_targets_all = mask_targets_all[:n_use]
    oldlp_all = oldlp_all[:n_use]
    adv_all = advantages[:n_use]
    token_mask_all = batch.token_mask[:n_use]
    cos_all = batch.cos[:, :n_use, : tokens_all.shape[1] - 1]
    sin_all = batch.sin[:, :n_use, : tokens_all.shape[1] - 1]
    if isinstance(batch.vision, VisionEmbeddings):
        v_tokens = batch.vision.tokens[:n_use]
        v_deep = batch.vision.deepstack
        v_deep = tuple(ds[:n_use] for ds in v_deep) if v_deep else ()
    else:
        v_tokens = batch.vision[:n_use]
        v_deep = ()

    metrics_accum = {}
    state = train_state
    coef = float(kl_coef)
    for _ in range(int(num_epochs)):
        for i in range(chunks):
            start = i * mb
            end = (i + 1) * mb
            sl = slice(start, end)
            if v_deep:
                v_mb = VisionEmbeddings(tokens=v_tokens[sl], deepstack=tuple(ds[sl] for ds in v_deep))
            else:
                v_mb = v_tokens[sl]
            state, m = _ppo_step(
                state,
                tokens_all[sl],
                mask_targets_all[sl, : tokens_all.shape[1] - 1],
                adv_all[sl],
                oldlp_all[sl, : tokens_all.shape[1] - 1],
                v_mb,
                token_mask_all[sl],
                cos_all[:, start:end],
                sin_all[:, start:end],
                coef,
            )

            # Adaptive KL if provided
            if kl_ctrl is not None:
                try:
                    approx_kl_val = float(jax.device_get(m.get('approx_kl', 0.0)))
                except Exception:
                    approx_kl_val = 0.0
                coef = kl_ctrl.update(approx_kl_val, coef)

            # Accumulate metrics
            m_host = jax.tree_util.tree_map(lambda x: float(jax.device_get(x)), m)
            for k, v in m_host.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

    # Average over total minibatches processed
    total_mb = chunks * int(num_epochs)
    for k in list(metrics_accum.keys()):
        metrics_accum[k] /= max(1, total_mb)
    metrics_accum['kl_coef'] = float(coef)
    return state, metrics_accum


__all__ = ["TrainerConfig", "collect", "update"]
