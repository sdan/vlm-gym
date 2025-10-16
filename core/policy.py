"""Tiny policy utilities: token logprobs and RoPE building.

This module provides a minimal, Linen-native wrapper around the model for
computing token-level logprobs and constructing mixed RoPE for Qwen3-VL. It
copies the relevant logic from existing trainers into a reusable API while
leaving current files untouched.
"""

from __future__ import annotations

from typing import Tuple, Union

import jax
import jax.numpy as jnp

from vlmrl.utils.train_state import TrainState
from vlmrl.models.qwen3vl.model import (
    Qwen3VLModel,
    VisionEmbeddings,
    get_rope_index,
    build_mrope,
)
from vlmrl.core.types import Batch


@jax.jit
def token_logprobs_vlm(
    train_state: TrainState,
    image_pad_id: int,
    tokens: jnp.ndarray,
    token_mask: jnp.ndarray,
    vision_embeds: Union[jnp.ndarray, VisionEmbeddings],
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    *,
    checkpoint: bool = False,
) -> jnp.ndarray:
    """Compute per-token logprobs for VLM given full tokens.

    This mirrors the inline helper previously defined in the GRPO trainer,
    calling model.forward_vlm with text_input=tokens[:, :-1].

    Args:
        train_state: TrainState containing params and model_def
        image_pad_id: vision pad/placeholder token id for injection
        tokens: [B, T] token ids
        token_mask: [B, T-1] mask over text targets
        vision_embeds: VisionEmbeddings or raw vision features
        cos/sin: RoPE caches for text_input=tokens[:, :-1]
        checkpoint: use jax.checkpoint around the forward call
    Returns:
        token_logprobs: [B, T-1] float32 log-probabilities for each next-token
    """
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

    if checkpoint:
        logits, _ = jax.checkpoint(call_with_params, prevent_cse=False)(train_state.params)
    else:
        logits, _ = call_with_params(train_state.params)

    logp = jax.nn.log_softmax(logits, axis=-1)  # [B, T-1, V]
    token_logp = jnp.sum(logp * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)
    return token_logp


def token_logprobs(
    train_state: TrainState,
    image_pad_id: int,
    batch: Batch,
    *,
    checkpoint: bool = False,
) -> jnp.ndarray:
    """Convenience wrapper using a typed Batch."""
    return token_logprobs_vlm(
        train_state,
        image_pad_id,
        batch.tokens,
        batch.token_mask,
        batch.vision,
        batch.cos,
        batch.sin,
        checkpoint=checkpoint,
    )


def build_rope(
    model: Qwen3VLModel,
    tokens: jnp.ndarray,
    grid: jnp.ndarray,
    token_mask: jnp.ndarray,
    *,
    max_chunk_size: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute mixed RoPE (cos, sin) in chunks to avoid OOM.

    This is a lifted copy of the helper from the minimal GRPO implementation.

    Args:
        model: Qwen3-VL model (provides spec and dtype)
        tokens: [B, T] full tokens
        grid: [B, num_images, 3] THW grid (or [1, 3] broadcastable)
        token_mask: [B, T-1] mask for text_input=tokens[:, :-1]
        max_chunk_size: number of batch rows per chunk
    Returns:
        cos, sin: RoPE caches matching text_input shape
    """
    text_input = tokens[:, :-1]
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
                attention_mask=chunk_mask,
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
            attention_mask=token_mask,
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


__all__ = [
    "token_logprobs_vlm",
    "token_logprobs",
    "build_rope",
]

