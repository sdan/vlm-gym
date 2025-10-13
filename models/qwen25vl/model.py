"""Qwen2.5-VL JAX model components.

Feature map:
- Generalized multi-head attention with GQA (num_heads vs num_kv_heads) and KV cache support.
- Multimodal RoPE via mRoPE builders and get_rope_index.
- Vision tower isolation plus image-pad embedding injection (see vision.py).
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from safetensors import safe_open


DType = jnp.dtype


def rms_norm(x: jax.Array, gamma: jax.Array, eps: float) -> jax.Array:
    # rmsnorm with tiny epsilon
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(variance + eps)
    return (gamma * x_norm).astype(x.dtype)


def rotate_half(x: jax.Array) -> jax.Array:
    # simple rotary quarter-turn
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_multimodal_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    rope_section: Sequence[int],
    unsqueeze_dim: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Apply rotary position embeddings to q and k.

    Expects pre-built cos/sin tensors matching the head_dim layout (either via
    ``build_text_rope`` for 1D text or ``build_mrope`` for multimodal tokens).
    """
    sections = tuple(int(x) for x in rope_section)
    if len(sections) == 0:
        raise ValueError("rope_section must be non-empty")
    total_dim = sum(sections)
    if cos.shape[-1] != total_dim * 2 or sin.shape[-1] != total_dim * 2:
        raise ValueError(
            f"Expected cos/sin last dim {total_dim * 2}, got cos={cos.shape[-1]}, sin={sin.shape[-1]}"
        )
    num_axes = cos.shape[0]
    if num_axes != len(sections):
        raise ValueError(
            f"rope_section length ({len(sections)}) must match cos/sin first dimension ({num_axes})"
        )

    def _reorder(table: jax.Array) -> jax.Array:
        idx = 0
        chunks: list[jax.Array] = []
        for _ in range(2):
            for sec in sections:
                next_idx = idx + sec
                chunks.append(table[..., idx:next_idx])
                idx = next_idx
        if not chunks:
            raise ValueError("Failed to split rotary tables; rope_section may be invalid.")
        gathered = [chunk[i % num_axes] for i, chunk in enumerate(chunks)]
        return jnp.concatenate(gathered, axis=-1)

    cos_flat = _reorder(cos).astype(q.dtype)
    sin_flat = _reorder(sin).astype(q.dtype)
    cos_embed = jnp.expand_dims(cos_flat, axis=unsqueeze_dim)
    sin_embed = jnp.expand_dims(sin_flat, axis=unsqueeze_dim)
    q_embed = q * cos_embed + rotate_half(q) * sin_embed
    k_embed = k * cos_embed + rotate_half(k) * sin_embed
    return q_embed, k_embed


def build_text_rope(
    positions: jax.Array,
    rope_section: Sequence[int],
    rope_theta: float,
    dtype: DType = jnp.bfloat16,
    *,
    rope_scaling_type: Optional[str] = None,
    rope_scaling_factor: Optional[float] = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute text RoPE sin/cos tensors for 1D positions.

    # classic rope for tokens
    """
    axes = len(tuple(int(x) for x in rope_section))
    if axes <= 0:
        raise ValueError("rope_section must contain at least one entry")
    pos_axes = jnp.broadcast_to(positions[None, ...], (axes,) + positions.shape)
    return build_mrope(
        pos_axes,
        rope_section,
        rope_theta,
        dtype=dtype,
        rope_scaling_type=rope_scaling_type,
        rope_scaling_factor=rope_scaling_factor,
    )


def build_mrope(
    position_ids_axes: jax.Array,
    rope_section: Sequence[int],
    rope_theta: float,
    dtype: DType = jnp.bfloat16,
    *,
    rope_scaling_type: Optional[str] = None,
    rope_scaling_factor: Optional[float] = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute multimodal RoPE sin/cos tensors for 3D indices.

    # separate axes then concat parts
    """

    if position_ids_axes.ndim != 3:
        raise ValueError("position_ids_axes must have shape (num_axes, batch, seq_len)")
    sections = tuple(int(x) for x in rope_section)
    num_axes = position_ids_axes.shape[0]
    if len(sections) != num_axes:
        raise ValueError(
            f"rope_section length ({len(sections)}) must equal number of axes ({num_axes})"
        )
    if any(sec <= 0 for sec in sections):
        raise ValueError("All rope_section entries must be positive")

    pos = position_ids_axes.astype(jnp.float32)
    if rope_scaling_factor is not None and rope_scaling_type in (None, "linear", "dynamic", "finetuned"):
        try:
            scale = float(rope_scaling_factor)
        except Exception:
            scale = 1.0
        if scale > 0.0:
            pos = pos / jnp.float32(scale)

    total_dim = sum(sections)
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, total_dim, dtype=jnp.float32) / float(total_dim)))
    freqs = jnp.einsum("sbn,k->sbnk", pos, inv_freq, precision=jax.lax.Precision.HIGHEST)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb).astype(dtype)
    sin = jnp.sin(emb).astype(dtype)
    return cos, sin


def get_rope_index(
    spatial_merge_size: int = 2,
    input_ids: Optional[jax.Array] = None,
    image_grid_thw: Optional[jax.Array] = None,
    video_grid_thw: Optional[jax.Array] = None,
    second_per_grid_ts: Optional[jax.Array] = None,
    attention_mask: Optional[jax.Array] = None,
    *,
    tokens_per_second: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    """Reproduce Qwen2.5-VL multimodal RoPE indexing logic in JAX.

    # linearize text + vision blocks
    """

    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    if input_ids is not None:
        batch, seq_len = input_ids.shape
    else:
        batch = attention_mask.shape[0] if attention_mask is not None else 1
        seq_len = attention_mask.shape[1] if attention_mask is not None else 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = jnp.ones_like(total_input_ids)

        position_ids = jnp.ones((3, batch, seq_len), dtype=total_input_ids.dtype)
        mrope_position_deltas: list[jax.Array] = []
        image_index = 0
        video_index = 0

        for i in range(batch):  # per-example pass
            ids = total_input_ids[i]
            mask = attention_mask[i]
            valid = ids[mask == 1]
            vision_start_indices = jnp.argwhere(valid == vision_start_token_id).flatten()
            vision_tokens = (
                valid[vision_start_indices + 1]
                if vision_start_indices.size > 0
                else jnp.array([], dtype=valid.dtype)
            )
            image_nums = int((vision_tokens == image_token_id).sum())
            video_nums = int((vision_tokens == video_token_id).sum())
            input_tokens = valid.tolist()

            llm_pos_ids_list: list[jax.Array] = []
            st = 0
            remain_images = image_nums
            remain_videos = video_nums

            for _ in range(image_nums + video_nums):  # walk chunks in order
                if remain_images > 0:
                    try:
                        ed_image = input_tokens.index(image_token_id, st)
                    except ValueError:
                        ed_image = len(input_tokens) + 1
                else:
                    ed_image = len(input_tokens) + 1
                if remain_videos > 0:
                    try:
                        ed_video = input_tokens.index(video_token_id, st)
                    except ValueError:
                        ed_video = len(input_tokens) + 1
                else:
                    ed_video = len(input_tokens) + 1

                use_image = ed_image < ed_video
                if use_image:
                    if image_grid_thw is None:
                        raise ValueError(
                            "image_grid_thw must be provided when image tokens are present"
                        )
                    t, h, w = [int(x) for x in image_grid_thw[image_index]]
                    second_per_grid_t = 0.0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    if video_grid_thw is None:
                        raise ValueError(
                            "video_grid_thw must be provided when video tokens are present"
                        )
                    t, h, w = [int(x) for x in video_grid_thw[video_index]]
                    if second_per_grid_ts is not None:
                        second_per_grid_t = float(second_per_grid_ts[video_index])
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = t
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                text_len = ed - st

                if llm_pos_ids_list:
                    st_idx = int(jnp.max(llm_pos_ids_list[-1])) + 1
                else:
                    st_idx = 0

                text_range = jnp.arange(text_len, dtype=valid.dtype)
                text_positions = jnp.tile(text_range.reshape(1, -1), (3, 1)) + st_idx
                llm_pos_ids_list.append(text_positions)

                range_tensor = jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1)  # time index
                expanded_range = jnp.tile(range_tensor, (1, llm_grid_h * llm_grid_w))
                time_tensor = (
                    expanded_range.astype(jnp.float32)
                    * float(second_per_grid_t)
                    * float(tokens_per_second)
                )
                t_index = time_tensor.astype(jnp.int32).reshape(-1)

                h_index = jnp.arange(llm_grid_h, dtype=jnp.int32).reshape(1, -1, 1)  # h index
                h_index = jnp.tile(h_index, (llm_grid_t, 1, llm_grid_w)).reshape(-1)
                w_index = jnp.arange(llm_grid_w, dtype=jnp.int32).reshape(1, 1, -1)  # w index
                w_index = jnp.tile(w_index, (llm_grid_t, llm_grid_h, 1)).reshape(-1)
                spatial = jnp.stack([t_index, h_index, w_index], axis=0)
                spatial = spatial + text_len + st_idx
                llm_pos_ids_list.append(spatial)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                if llm_pos_ids_list:
                    st_idx = int(jnp.max(llm_pos_ids_list[-1])) + 1
                else:
                    st_idx = 0
                text_len = len(input_tokens) - st
                text_range = jnp.arange(text_len, dtype=valid.dtype)
                text_positions = jnp.tile(text_range.reshape(1, -1), (3, 1)) + st_idx
                llm_pos_ids_list.append(text_positions)

            llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            sel = jnp.where(mask == 1)[0]
            position_ids = position_ids.at[:, i, sel].set(llm_positions)
            delta = jnp.array(int(jnp.max(llm_positions)) + 1 - seq_len, dtype=valid.dtype)
            mrope_position_deltas.append(delta)

        mrope_position_deltas = jnp.stack(mrope_position_deltas).reshape(batch, 1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        mask = attention_mask.astype(jnp.int32)
        position_ids = jnp.cumsum(mask, axis=-1) - 1
        position_ids = jnp.where(mask == 0, 1, position_ids)
        position_ids = jnp.tile(position_ids[None, ...], (3, 1, 1))
        max_position_ids = position_ids.max(axis=0)
        max_position_ids = max_position_ids.max(axis=-1, keepdims=True)
        mrope_position_deltas = max_position_ids + 1 - seq_len
        return position_ids, mrope_position_deltas

    position_ids = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, 1, -1)
    position_ids = jnp.tile(position_ids, (3, batch, 1))
    mrope_position_deltas = jnp.zeros((batch, 1), dtype=jnp.int32)
    return position_ids, mrope_position_deltas


def _build_rope_from_positions(
    positions: jax.Array,
    rope_section: tuple[int, ...],
    rope_theta: float,
    dtype: DType,
    deltas: Optional[jax.Array] = None,
    *,
    rope_scaling_type: Optional[str] = None,
    rope_scaling_factor: Optional[float] = None,
) -> tuple[jax.Array, jax.Array]:
    if deltas is None:
        return build_text_rope(
            positions,
            rope_section,
            rope_theta,
            dtype=dtype,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
        )
    shifted = positions + deltas
    axes = len(rope_section)
    pos3 = jnp.broadcast_to(shifted[None, :, :], (axes, shifted.shape[0], shifted.shape[1]))
    return build_mrope(
        pos3,
        rope_section,
        rope_theta,
        dtype=dtype,
        rope_scaling_type=rope_scaling_type,
        rope_scaling_factor=rope_scaling_factor,
    )


@dataclass
class TextBackboneSpec:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta: float
    rope_section: Sequence[int]
    rope_scaling_type: Optional[str]
    rope_scaling_factor: Optional[float]
    rms_norm_eps: float
    vocab_size: int


@dataclass
class VisionBackboneSpec:
    hidden_size: int
    out_hidden_size: int
    depth: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    window_size: int
    in_channels: int
    tokens_per_second: int
    fullatt_block_indexes: Sequence[int]


@dataclass
class Qwen25VLSpec:
    text: TextBackboneSpec
    vision: Optional[VisionBackboneSpec]
    pad_token_id: Optional[int]
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float
    dtype: DType = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        gamma = self.param("weight", nn.initializers.ones, (self.hidden_size,), self.dtype)
        return rms_norm(x, gamma, self.eps)


class FeedForward(nn.Module):
    hidden_size: int
    intermediate_size: int
    dtype: DType = jnp.bfloat16
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        gate = nn.Dense(
            self.intermediate_size, use_bias=self.use_bias, dtype=self.dtype, name="gate_proj"
        )(x)
        up = nn.Dense(
            self.intermediate_size, use_bias=self.use_bias, dtype=self.dtype, name="up_proj"
        )(x)
        down = nn.Dense(
            self.hidden_size, use_bias=self.use_bias, dtype=self.dtype, name="down_proj"
        )(nn.silu(gate) * up)
        return down


class MultiHeadAttention(nn.Module):
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_section: Optional[Sequence[int]] = None
    dtype: DType = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional["KVCache"] = None,
        layer_id: Optional[int] = None,
        update_lengths: bool = False,
    ) -> tuple[jax.Array, Optional["KVCache"]]:
        q = nn.Dense(
            self.num_heads * self.head_dim, use_bias=True, dtype=self.dtype, name="q_proj"
        )(x)
        k = nn.Dense(
            self.num_kv_heads * self.head_dim, use_bias=True, dtype=self.dtype, name="k_proj"
        )(x)
        v = nn.Dense(
            self.num_kv_heads * self.head_dim, use_bias=True, dtype=self.dtype, name="v_proj"
        )(x)

        batch, seqlen, _ = q.shape
        q = q.reshape(batch, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(batch, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seqlen, self.num_kv_heads, self.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_section)  # rope here

        if mask is not None:
            if jnp.issubdtype(mask.dtype, jnp.floating):
                key_mask = mask
            else:
                key_mask = mask.astype(jnp.float32)
            cache_lengths = key_mask.sum(axis=-1).astype(jnp.int32)  # valid tokens
            broadcast_mask = key_mask[:, None, :, None].astype(k.dtype)
            k = k * broadcast_mask
            v = v * broadcast_mask
        else:
            cache_lengths = jnp.full((q.shape[0],), q.shape[2], dtype=jnp.int32)

        if cache is not None:
            start_positions = cache.lengths  # append to kv cache
            full_k, full_v, cache = cache.update(layer_id, k, v, start_positions, cache_lengths)
            k = full_k
            v = full_v
            effective_lengths = start_positions + cache_lengths
            if update_lengths:
                cache = cache.replace(lengths=effective_lengths)
        else:
            effective_lengths = cache_lengths

        history_mask = (
            (jnp.arange(k.shape[2])[None, :] < effective_lengths[:, None]).astype(jnp.float32)
        )
        # Grouped-query attention without explicit repeat
        if self.num_heads != self.num_kv_heads:
            # Reshape for grouped attention: group queries with their kv heads
            repeats = self.num_heads // self.num_kv_heads
            q_grouped = q.reshape(batch, self.num_kv_heads, repeats, q.shape[2], self.head_dim)
            # k stays as (batch, kv_heads, k_len, dim) - no need to expand yet
            
            # Compute attention scores with einsum that handles the broadcasting
            attn_scores = jnp.einsum(
                "bhgqd,bhkd->bhgqk",
                q_grouped.astype(jnp.float32),
                k.astype(jnp.float32),
            ) * (self.head_dim ** -0.5)
            # Reshape back to (batch, num_heads, q_len, k_len)
            attn_scores = attn_scores.reshape(batch, self.num_heads, q.shape[2], k.shape[2])
        else:
            attn_scores = jnp.einsum(
                "bhqd,bhkd->bhqk",
                q.astype(jnp.float32),
                k.astype(jnp.float32),
            ) * (self.head_dim ** -0.5)
        attn_scores = attn_scores + (1.0 - history_mask)[:, None, None, :].astype(jnp.float32) * -1e9  # mask pad
        q_len = attn_scores.shape[2]
        if q_len > 1:
            k_len = attn_scores.shape[3]
            causal = jnp.tril(jnp.ones((q_len, k_len), dtype=jnp.float32))  # causal triangle
            attn_scores = attn_scores + (1.0 - causal)[None, None, :, :] * -1e9
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        
        # Apply attention weights with grouped-query pattern if needed
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            # Reshape weights for grouped computation
            attn_weights_grouped = attn_weights.reshape(batch, self.num_kv_heads, repeats, q_len, -1)
            # Compute output with einsum that handles the broadcasting
            attn_output = jnp.einsum(
                "bhgqk,bhkd->bhgqd",
                attn_weights_grouped,
                v.astype(jnp.float32),
            )
            # Reshape back to (batch, num_heads, q_len, head_dim)
            attn_output = attn_output.reshape(batch, self.num_heads, q_len, self.head_dim).astype(self.dtype)
        else:
            attn_output = jnp.einsum(
                "bhqk,bhkd->bhqd",
                attn_weights,
                v.astype(jnp.float32),
            ).astype(self.dtype)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3)).reshape(batch, seqlen, -1)
        out = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype, name="o_proj")(
            attn_output
        )
        return out, cache


class DecoderBlock(nn.Module):
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rope_section: Sequence[int]
    eps: float
    dtype: DType = jnp.bfloat16

    def setup(self) -> None:
        self.input_norm = RMSNorm(self.hidden_size, self.eps, self.dtype)
        self.post_norm = RMSNorm(self.hidden_size, self.eps, self.dtype)
        self.attn = MultiHeadAttention(
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            rope_section=self.rope_section,
            dtype=self.dtype,
        )
        self.mlp = FeedForward(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array],
        cache: Optional["KVCache"],
        layer_id: int,
        update_lengths: bool = False,
    ) -> tuple[jax.Array, Optional["KVCache"]]:
        attn_out, cache = self.attn(
            self.input_norm(hidden_states),
            cos,
            sin,
            mask,
            cache,
            layer_id,
            update_lengths=update_lengths,
        )
        hidden_states = hidden_states + attn_out  # res
        hidden_states = hidden_states + self.mlp(self.post_norm(hidden_states))  # res
        return hidden_states, cache


class KVCache(flax.struct.PyTreeNode):
    keys: jax.Array
    values: jax.Array
    lengths: jax.Array

    @classmethod
    def init(
        cls,
        batch: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_len: int,
        dtype: DType,
    ) -> "KVCache":
        # pre-allocate [layers, batch, heads, T, D]
        keys = jnp.zeros((num_layers, batch, num_heads, max_len, head_dim), dtype=dtype)
        values = jnp.zeros((num_layers, batch, num_heads, max_len, head_dim), dtype=dtype)
        lengths = jnp.zeros((batch,), dtype=jnp.int32)
        return cls(keys=keys, values=values, lengths=lengths)

    def update(
        self,
        layer_id: int,
        k: jax.Array,
        v: jax.Array,
        start_positions: jax.Array,
        chunk_lengths: jax.Array,
    ) -> tuple[jax.Array, jax.Array, "KVCache"]:
        """Append new keys/values for a layer and return full cached tensors."""

        def _update_one(cache_k, cache_v, new_k, new_v, start, chunk_len):
            # write slice into ring buffer
            chunk_len = jnp.asarray(chunk_len, dtype=jnp.int32)
            start = jnp.asarray(start, dtype=jnp.int32)
            mask = (jnp.arange(new_k.shape[1]) < chunk_len)[None, :, None]
            new_k = new_k * mask
            new_v = new_v * mask
            updated_k = jax.lax.dynamic_update_slice(cache_k, new_k, (0, start, 0))
            updated_v = jax.lax.dynamic_update_slice(cache_v, new_v, (0, start, 0))
            return updated_k, updated_v

        layer_keys = self.keys[layer_id]
        layer_values = self.values[layer_id]
        updated = jax.vmap(
            _update_one,
            in_axes=(0, 0, 0, 0, 0, 0),
            out_axes=(0, 0),
        )(layer_keys, layer_values, k, v, start_positions, chunk_lengths)

        new_layer_keys, new_layer_values = updated
        keys = self.keys.at[layer_id].set(new_layer_keys)
        values = self.values.at[layer_id].set(new_layer_values)
        cache = self.replace(keys=keys, values=values)
        return new_layer_keys, new_layer_values, cache


class Qwen25VLModel(nn.Module):
    spec: Qwen25VLSpec
    dtype: DType = jnp.bfloat16

    def setup(self) -> None:
        text = self.spec.text
        embed_init = nn.initializers.normal(stddev=0.02)
        self.embed = nn.Embed(text.vocab_size, text.hidden_size, embedding_init=embed_init, dtype=self.dtype)
        # stack N decoder blocks
        self.layers = [
            DecoderBlock(
                hidden_size=text.hidden_size,
                num_heads=text.num_attention_heads,
                num_kv_heads=text.num_key_value_heads,
                head_dim=text.head_dim,
                intermediate_size=text.intermediate_size,
                rope_section=tuple(text.rope_section),
                eps=text.rms_norm_eps,
                dtype=self.dtype,
            )
            for _ in range(text.num_hidden_layers)
        ]
        self.final_norm = RMSNorm(text.hidden_size, text.rms_norm_eps, self.dtype)
        self.lm_head = nn.Dense(text.vocab_size, use_bias=False, dtype=jnp.float32)  # logits in fp32
        if self.spec.vision is not None:
            from .vision import Qwen25VisionTransformer

            self.visual = Qwen25VisionTransformer(self.spec.vision)  # optional vision tower
        else:
            self.visual = None

    def _decode_from_hidden(
        self,
        hidden: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[jax.Array, Optional[KVCache]]:
        new_cache = cache
        last_layer_idx = len(self.layers) - 1
        for layer_id, layer in enumerate(self.layers):
            hidden, new_cache = layer(
                hidden,
                cos,
                sin,
                mask,
                new_cache,
                layer_id,
                update_lengths=bool(cache is not None and layer_id == last_layer_idx),  # bump cache len at last layer
            )
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden.astype(jnp.float32))  # keep logits in fp32
        return logits, new_cache

    def forward_text(
        self,
        tokens: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[jax.Array, Optional[KVCache]]:
        hidden = self.embed(tokens)
        return self._decode_from_hidden(hidden, cos, sin, mask, cache)

    def forward_vlm(
        self,
        tokens: jax.Array,
        vision_embeds: jax.Array,
        image_pad_id: int,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[jax.Array, Optional[KVCache]]:
        hidden = self.embed(tokens)
        if vision_embeds.ndim == 2:
            vision_embeds = vision_embeds[None, ...]
        if vision_embeds.ndim != 3:
            raise ValueError("vision_embeds must have shape (batch, num_tokens, hidden)")
        batch = hidden.shape[0]
        if vision_embeds.shape[0] not in (1, batch):
            raise ValueError("vision_embeds batch dimension must be 1 or match tokens batch")
        if vision_embeds.shape[0] == 1 and batch > 1:
            vision_embeds = jnp.tile(vision_embeds, (batch, 1, 1))

        def _inject(batch_hidden, batch_tokens, batch_vis):
            # overlay vision embeddings at <|image_pad|> positions
            # image_pad_id is a special placeholder token id
            num_vision = batch_vis.shape[0]
            pad_positions = jnp.where(
                batch_tokens == jnp.int32(image_pad_id), size=num_vision, fill_value=-1
            )[0]
            valid = pad_positions >= 0
            pad_positions = jnp.where(valid, pad_positions, 0)
            updates = jnp.where(valid[:, None], batch_vis.astype(batch_hidden.dtype), batch_hidden[pad_positions])
            batch_hidden = batch_hidden.at[pad_positions].set(updates)
            return batch_hidden

        hidden = jax.vmap(_inject)(hidden, tokens, vision_embeds)
        return self._decode_from_hidden(hidden, cos, sin, mask, cache)

    def encode_vision(self, pixel_values: jax.Array, grid_thw: jax.Array) -> jax.Array:
        if self.visual is None:
            raise ValueError("Vision backbone not configured for this model")
        return self.visual(pixel_values, grid_thw)

    def decode_step(
        self,
        token: jax.Array,
        cache: KVCache,
        rope_deltas: Optional[jax.Array],
        mask: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, KVCache]:
        if token.ndim != 1:
            raise ValueError("token must have shape (batch,)")
        positions = cache.lengths[:, None]  # next token positions
        batch = positions.shape[0]
        axes = len(tuple(int(x) for x in self.spec.text.rope_section))
        if axes <= 0:
            raise ValueError("Model spec must define a non-empty rope_section")
        base_positions = jnp.broadcast_to(positions[None, :, :], (axes, batch, positions.shape[1]))
        if rope_deltas is not None:
            rope_offsets = rope_deltas.astype(jnp.int32)[None, :, :]
        else:
            rope_offsets = jnp.zeros((axes, batch, 1), dtype=jnp.int32)
        position_ids_axes = base_positions + rope_offsets
        cos, sin = build_mrope(
            position_ids_axes,
            tuple(self.spec.text.rope_section),
            self.spec.text.rope_theta,
            dtype=self.dtype,
            rope_scaling_type=self.spec.text.rope_scaling_type,
            rope_scaling_factor=self.spec.text.rope_scaling_factor,
        )
        step_tokens = token[:, None]
        if mask is None:
            mask = jnp.ones((step_tokens.shape[0], 1), dtype=jnp.int32)
        logits, new_cache = self.forward_text(step_tokens, cos, sin, mask=mask, cache=cache)
        return logits[:, -1, :], new_cache

    def __call__(
        self,
        tokens: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[jax.Array, Optional[KVCache]]:
        return self.forward_text(tokens, cos, sin, mask=mask, cache=cache)


def _load_hf_config(hf_dir: str) -> dict[str, Any]:
    with open(f"{hf_dir}/config.json") as f:
        return json.load(f)


def spec_from_config(cfg: dict[str, Any]) -> Qwen25VLSpec:
    # Map HF config to our internal spec
    text_cfg = cfg.get("text_config", cfg)
    rope_cfg = text_cfg.get("rope_scaling", cfg.get("rope_scaling", {}))
    # Compute common head_dim once; needed for RoPE section validation
    _head_dim = text_cfg["hidden_size"] // text_cfg["num_attention_heads"]
    vision_cfg = cfg.get("vision_config")

    # Parse optional rope scaling type/factor
    rope_scaling_type = None
    rope_scaling_factor = None
    if isinstance(rope_cfg, dict):
        rope_scaling_type = rope_cfg.get("type")
        rope_scaling_factor = rope_cfg.get("factor", rope_cfg.get("finetuned_factor"))

    # Validate/derive rope_section
    raw_section = (rope_cfg.get("mrope_section", None) if isinstance(rope_cfg, dict) else None)
    rope_section: list[int]
    if raw_section is None:
        # If a vision tower is configured, mRoPE sections are checkpoint-dependent.
        # Fail loudly rather than guessing a split that could corrupt angles.
        if vision_cfg is not None:
            raise ValueError(
                "Missing rope_scaling.mrope_section for vision model; expected 3-tuple summing to head_dim//2."
            )
        # Text-only: fallback to classic RoPE with half-dim
        rope_section = [_head_dim // 2]
    else:
        # Normalize and validate
        rope_section = [int(x) for x in raw_section]
        if vision_cfg is not None and len(rope_section) != 3:
            raise ValueError("rope_scaling.mrope_section must have 3 entries (t,h,w) for vision models")
        expected = _head_dim // 2
        if sum(rope_section) != expected:
            raise ValueError(
                f"Sum of rope_scaling.mrope_section must equal head_dim//2 ({expected}); got {sum(rope_section)}"
            )
    text = TextBackboneSpec(
        hidden_size=text_cfg["hidden_size"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_hidden_layers=text_cfg["num_hidden_layers"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        head_dim=_head_dim,
        intermediate_size=text_cfg["intermediate_size"],
        rope_theta=text_cfg.get("rope_theta", cfg.get("rope_theta", 10000.0)),
        rope_section=rope_section,
        rope_scaling_type=rope_scaling_type,
        rope_scaling_factor=(float(rope_scaling_factor) if rope_scaling_factor is not None else None),
        rms_norm_eps=text_cfg["rms_norm_eps"],
        vocab_size=text_cfg["vocab_size"],
    )
    vision = None
    if vision_cfg is not None:
        vision = VisionBackboneSpec(
            hidden_size=vision_cfg["hidden_size"],
            out_hidden_size=vision_cfg["out_hidden_size"],
            depth=vision_cfg["depth"],
            num_heads=vision_cfg["num_heads"],
            intermediate_size=vision_cfg["intermediate_size"],
            patch_size=vision_cfg.get("patch_size", vision_cfg.get("spatial_patch_size")),
            temporal_patch_size=vision_cfg["temporal_patch_size"],
            spatial_merge_size=vision_cfg["spatial_merge_size"],
            window_size=vision_cfg["window_size"],
            in_channels=vision_cfg.get("in_channels", vision_cfg.get("in_chans", 3)),
            tokens_per_second=vision_cfg.get("tokens_per_second", 1),
            fullatt_block_indexes=vision_cfg.get("fullatt_block_indexes", []),
        )
    return Qwen25VLSpec(
        text=text,
        vision=vision,
        pad_token_id=cfg.get("pad_token_id"),
        bos_token_id=cfg.get("bos_token_id"),
        eos_token_id=cfg.get("eos_token_id"),
    )


# Regex map: HF torch param names -> Flax tree
_TEXT_KEY_RULES = {
    r"model\.language_model\.model\.embed_tokens\.weight": "embed/embedding",
    r"model\.language_model\.embed_tokens\.weight": "embed/embedding",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"layers_\1/attn/q_proj/kernel",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"layers_\1/attn/q_proj/kernel",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": r"layers_\1/attn/q_proj/bias",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": r"layers_\1/attn/q_proj/bias",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers_\1/attn/k_proj/kernel",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers_\1/attn/k_proj/kernel",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": r"layers_\1/attn/k_proj/bias",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": r"layers_\1/attn/k_proj/bias",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"layers_\1/attn/v_proj/kernel",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"layers_\1/attn/v_proj/kernel",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": r"layers_\1/attn/v_proj/bias",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": r"layers_\1/attn/v_proj/bias",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"layers_\1/attn/o_proj/kernel",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"layers_\1/attn/o_proj/kernel",
    r"model\.language_model\.model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"layers_\1/mlp/gate_proj/kernel",
    r"model\.language_model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"layers_\1/mlp/gate_proj/kernel",
    r"model\.language_model\.model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"layers_\1/mlp/up_proj/kernel",
    r"model\.language_model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"layers_\1/mlp/up_proj/kernel",
    r"model\.language_model\.model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"layers_\1/mlp/down_proj/kernel",
    r"model\.language_model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"layers_\1/mlp/down_proj/kernel",
    r"model\.language_model\.model\.layers\.([0-9]+)\.input_layernorm\.weight": r"layers_\1/input_norm/weight",
    r"model\.language_model\.layers\.([0-9]+)\.input_layernorm\.weight": r"layers_\1/input_norm/weight",
    r"model\.language_model\.model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"layers_\1/post_norm/weight",
    r"model\.language_model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"layers_\1/post_norm/weight",
    r"model\.language_model\.model\.norm\.weight": "final_norm/weight",
    r"model\.language_model\.norm\.weight": "final_norm/weight",
    r"lm_head\.weight": "lm_head/kernel",
    r"model\.embed_tokens\.weight": "embed/embedding",
    r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"layers_\1/attn/q_proj/kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": r"layers_\1/attn/q_proj/bias",
    r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers_\1/attn/k_proj/kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": r"layers_\1/attn/k_proj/bias",
    r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"layers_\1/attn/v_proj/kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": r"layers_\1/attn/v_proj/bias",
    r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"layers_\1/attn/o_proj/kernel",
    r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"layers_\1/mlp/gate_proj/kernel",
    r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"layers_\1/mlp/up_proj/kernel",
    r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"layers_\1/mlp/down_proj/kernel",
    r"model\.layers\.([0-9]+)\.input_layernorm\.weight": r"layers_\1/input_norm/weight",
    r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"layers_\1/post_norm/weight",
    r"model\.norm\.weight": "final_norm/weight",
}


# Regex map for vision tower params
_VISION_KEY_RULES = {
    r"model\.visual\.blocks\.([0-9]+)\.norm1\.weight": r"visual/blocks_\1/norm1/weight",
    r"visual\.blocks\.([0-9]+)\.norm1\.weight": r"visual/blocks_\1/norm1/weight",
    r"model\.visual\.blocks\.([0-9]+)\.norm2\.weight": r"visual/blocks_\1/norm2/weight",
    r"visual\.blocks\.([0-9]+)\.norm2\.weight": r"visual/blocks_\1/norm2/weight",
    r"model\.visual\.blocks\.([0-9]+)\.attn\.qkv\.weight": r"visual/blocks_\1/attn/qkv/kernel",
    r"visual\.blocks\.([0-9]+)\.attn\.qkv\.weight": r"visual/blocks_\1/attn/qkv/kernel",
    r"model\.visual\.blocks\.([0-9]+)\.attn\.qkv\.bias": r"visual/blocks_\1/attn/qkv/bias",
    r"visual\.blocks\.([0-9]+)\.attn\.qkv\.bias": r"visual/blocks_\1/attn/qkv/bias",
    r"model\.visual\.blocks\.([0-9]+)\.attn\.proj\.weight": r"visual/blocks_\1/attn/proj/kernel",
    r"visual\.blocks\.([0-9]+)\.attn\.proj\.weight": r"visual/blocks_\1/attn/proj/kernel",
    r"model\.visual\.blocks\.([0-9]+)\.attn\.proj\.bias": r"visual/blocks_\1/attn/proj/bias",
    r"visual\.blocks\.([0-9]+)\.attn\.proj\.bias": r"visual/blocks_\1/attn/proj/bias",
    r"model\.visual\.blocks\.([0-9]+)\.mlp\.gate_proj\.weight": r"visual/blocks_\1/mlp/gate_proj/kernel",
    r"visual\.blocks\.([0-9]+)\.mlp\.gate_proj\.weight": r"visual/blocks_\1/mlp/gate_proj/kernel",
    r"model\.visual\.blocks\.([0-9]+)\.mlp\.gate_proj\.bias": r"visual/blocks_\1/mlp/gate_proj/bias",
    r"visual\.blocks\.([0-9]+)\.mlp\.gate_proj\.bias": r"visual/blocks_\1/mlp/gate_proj/bias",
    r"model\.visual\.blocks\.([0-9]+)\.mlp\.up_proj\.weight": r"visual/blocks_\1/mlp/up_proj/kernel",
    r"visual\.blocks\.([0-9]+)\.mlp\.up_proj\.weight": r"visual/blocks_\1/mlp/up_proj/kernel",
    r"model\.visual\.blocks\.([0-9]+)\.mlp\.up_proj\.bias": r"visual/blocks_\1/mlp/up_proj/bias",
    r"visual\.blocks\.([0-9]+)\.mlp\.up_proj\.bias": r"visual/blocks_\1/mlp/up_proj/bias",
    r"model\.visual\.blocks\.([0-9]+)\.mlp\.down_proj\.weight": r"visual/blocks_\1/mlp/down_proj/kernel",
    r"visual\.blocks\.([0-9]+)\.mlp\.down_proj\.weight": r"visual/blocks_\1/mlp/down_proj/kernel",
    r"model\.visual\.blocks\.([0-9]+)\.mlp\.down_proj\.bias": r"visual/blocks_\1/mlp/down_proj/bias",
    r"visual\.blocks\.([0-9]+)\.mlp\.down_proj\.bias": r"visual/blocks_\1/mlp/down_proj/bias",
    r"model\.visual\.merger\.ln_q\.weight": "visual/merger/norm/weight",
    r"visual\.merger\.ln_q\.weight": "visual/merger/norm/weight",
    r"model\.visual\.merger\.mlp\.0\.weight": "visual/merger/linear1/kernel",
    r"visual\.merger\.mlp\.0\.weight": "visual/merger/linear1/kernel",
    r"model\.visual\.merger\.mlp\.0\.bias": "visual/merger/linear1/bias",
    r"visual\.merger\.mlp\.0\.bias": "visual/merger/linear1/bias",
    r"model\.visual\.merger\.mlp\.2\.weight": "visual/merger/linear2/kernel",
    r"visual\.merger\.mlp\.2\.weight": "visual/merger/linear2/kernel",
    r"model\.visual\.merger\.mlp\.2\.bias": "visual/merger/linear2/bias",
    r"visual\.merger\.mlp\.2\.bias": "visual/merger/linear2/bias",
}


def _torch_key_to_flax(key: str) -> Optional[str]:
    for mapping in (_TEXT_KEY_RULES, _VISION_KEY_RULES):
        for pattern, target in mapping.items():
            if re.match(pattern, key):
                return re.sub(pattern, target, key)
    return None


def create_model_from_hf(hf_dir: str) -> tuple[Qwen25VLModel, dict[str, Any]]:
    cfg = _load_hf_config(hf_dir)
    spec = spec_from_config(cfg)
    model = Qwen25VLModel(spec)

    dummy_ids = jnp.zeros((1, 1), dtype=jnp.int32)
    rope_axes = len(tuple(int(x) for x in spec.text.rope_section))
    dummy_cos = jnp.zeros((rope_axes, 1, 1, spec.text.head_dim), dtype=model.dtype)
    dummy_sin = jnp.zeros((rope_axes, 1, 1, spec.text.head_dim), dtype=model.dtype)
    text_vars = model.init(jax.random.PRNGKey(0), dummy_ids, dummy_cos, dummy_sin)
    param_dict = flax.core.unfreeze(text_vars["params"])

    if spec.vision is not None:
        patch_volume = (
            spec.vision.in_channels
            * spec.vision.temporal_patch_size
            * spec.vision.patch_size
            * spec.vision.patch_size
        )
        merge = spec.vision.spatial_merge_size
        num_tokens = merge * merge
        dummy_pixels = jnp.zeros((num_tokens, patch_volume), dtype=model.dtype)
        dummy_grid = jnp.array([[1, merge, merge]], dtype=jnp.int32)
        vision_vars = model.init(
            jax.random.PRNGKey(1),
            dummy_pixels,
            dummy_grid,
            method=model.encode_vision,
        )
        param_dict.update(flax.core.unfreeze(vision_vars["params"]))

    safetensor_paths = glob.glob(f"{hf_dir}/*.safetensors")  # hf shard files
    if not safetensor_paths:
        raise FileNotFoundError(f"No safetensors files found in {hf_dir}")

    for path in safetensor_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in (
                    "visual.patch_embed.proj.weight",
                    "model.visual.patch_embed.proj.weight",
                ):
                    # conv->linear flatten
                    tensor = f.get_tensor(key).float().numpy()
                    tensor = tensor.reshape(tensor.shape[0], -1).T
                    current = (
                        param_dict.setdefault("visual", {})
                        .setdefault("patch_embed", {})
                        .setdefault("proj", {})
                    )
                    current["kernel"] = tensor
                    continue

                target = _torch_key_to_flax(key)  # regex-based map
                if target is None:
                    continue
                tensor = f.get_tensor(key).float().numpy()
                jax_key_list = target.split("/")
                jax_param = param_dict
                while len(jax_key_list) > 0:
                    jax_key = jax_key_list.pop(0)
                    if len(jax_key_list) == 0:
                        if "kernel" in jax_key:  # transpose for Dense
                            tensor = tensor.T
                        jax_param[jax_key] = tensor
                    else:
                        jax_param = jax_param[jax_key]

    return model, flax.core.freeze(param_dict)


def create_model_from_ckpt(ckpt_dir: str) -> tuple[Qwen25VLModel, dict[str, Any]]:
    from vlmrl.utils.checkpoint import Checkpoint

    cfg = _load_hf_config(ckpt_dir)
    spec = spec_from_config(cfg)
    model = Qwen25VLModel(spec)
    ckpt = Checkpoint(f"{ckpt_dir}/params.pkl", parallel=False)
    params = ckpt.load_as_dict()["params"]
    return model, params


__all__ = [
    "Qwen25VLModel",
    "KVCache",
    "apply_multimodal_rotary_pos_emb",
    "build_text_rope",
    "build_mrope",
    "get_rope_index",
    "spec_from_config",
    "create_model_from_hf",
    "create_model_from_ckpt",
]
