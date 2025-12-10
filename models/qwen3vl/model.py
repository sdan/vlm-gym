"""Qwen3-VL JAX model components

https://github.com/huggingface/transformers/blob/caa14e7dabb086f167c14b7eecadc2ba9db25eb6/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py
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
from flax import struct
from safetensors import safe_open


DType = jnp.dtype


@struct.dataclass
class VisionEmbeddings:
    tokens: jax.Array
    deepstack: tuple[jax.Array, ...] = ()

    @classmethod
    def concatenate(cls, embeds: Sequence["VisionEmbeddings"]) -> "VisionEmbeddings":
        if not embeds:
            return cls(tokens=jnp.zeros((0, 0), dtype=jnp.float32), deepstack=())
        base_deepstack_len = len(embeds[0].deepstack)
        tokens = jnp.concatenate([jnp.asarray(e.tokens) for e in embeds], axis=0)
        for emb in embeds[1:]:
            if len(emb.deepstack) != base_deepstack_len:
                raise ValueError("All VisionEmbeddings must have the same number of DeepStack levels to concatenate.")
        deepstack = tuple(
            jnp.concatenate([jnp.asarray(e.deepstack[idx]) for e in embeds], axis=0)
            for idx in range(base_deepstack_len)
        )
        return cls(tokens=tokens, deepstack=deepstack)

    def cast(self, dtype: DType) -> "VisionEmbeddings":
        tokens = jnp.asarray(self.tokens, dtype=dtype)
        deepstack = tuple(jnp.asarray(feat, dtype=dtype) for feat in self.deepstack)
        return VisionEmbeddings(tokens=tokens, deepstack=deepstack)

    def with_batch_dim(self, batch: int) -> "VisionEmbeddings":
        """Ensure tokens/deepstack include batch dimension of size `batch` or 1."""
        tokens = self.tokens
        if tokens.ndim == 2:
            tokens = tokens[None, ...]
        if tokens.shape[0] not in (1, batch):
            raise ValueError(
                f"Vision tokens batch dimension must be 1 or match input batch; got {tokens.shape[0]}, expected {batch}."
            )
        if tokens.shape[0] == 1 and batch > 1:
            tokens = jnp.tile(tokens, (batch, 1, 1))

        deepstack = []
        for feat in self.deepstack:
            if feat.ndim == 2:
                feat = feat[None, ...]
            if feat.shape[0] not in (1, batch):
                raise ValueError(
                    f"DeepStack batch dimension must be 1 or match input batch; got {feat.shape[0]}, expected {batch}."
                )
            if feat.shape[0] == 1 and batch > 1:
                feat = jnp.tile(feat, (batch, 1, 1))
            deepstack.append(feat)

        return VisionEmbeddings(tokens=tokens, deepstack=tuple(deepstack))


def rms_norm(x: jax.Array, gamma: jax.Array, eps: float) -> jax.Array:
    # rmsnorm with tiny epsilon
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(variance + eps)
    return (gamma * x_norm).astype(x.dtype)


class LoRADense(nn.Module):
    """Dense layer with optional LoRA adapter.

    Parameters are structured to be compatible with existing checkpoint mappings:
    - base weights live under `kernel` (and `bias` when enabled), just like nn.Dense
    - LoRA adapters introduce `lora_A` and `lora_B` params when rank > 0
    """

    features: int
    use_bias: bool = True
    dtype: DType = jnp.bfloat16
    rank: int = 0
    alpha: float = 1.0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_features = int(x.shape[-1])
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (in_features, self.features),
            self.dtype,
        )
        y = jnp.einsum("...d,df->...f", x.astype(self.dtype), kernel)
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.dtype)
            y = y + bias

        if int(self.rank) > 0:
            # LoRA path: y += scale * (x @ A @ B)
            A = self.param(
                "lora_A",
                nn.initializers.normal(stddev=0.02),
                (in_features, int(self.rank)),
                self.dtype,
            )
            B = self.param(
                "lora_B", nn.initializers.zeros, (int(self.rank), self.features), self.dtype
            )
            scale = jnp.asarray(self.alpha / float(max(1, int(self.rank))), dtype=jnp.float32)
            lora = jnp.einsum("...d,dr->...r", x.astype(jnp.float32), A.astype(jnp.float32))
            lora = jnp.einsum("...r,rf->...f", lora, B.astype(jnp.float32)).astype(self.dtype)
            y = y + (scale.astype(self.dtype) * lora)
        return y


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

    Supports interleaved mRoPE where only part of head_dim is rotated.
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

    # Handle interleaved mRoPE: only rotate part of head_dim
    rope_dim = total_dim * 2
    head_dim = q.shape[-1]
    if rope_dim > head_dim:
        # Interleaved: cos/sin are for more dims than head, rotate only first total_dim of head
        rotated_dim = total_dim
        q_rot = q[..., :rotated_dim]
        q_pass = q[..., rotated_dim:]
        k_rot = k[..., :rotated_dim]
        k_pass = k[..., rotated_dim:]
        # Take only the dimensions needed for the rotated part
        cos_rot = cos_embed[..., :rotated_dim]
        sin_rot = sin_embed[..., :rotated_dim]
        q_rot_embed = q_rot * cos_rot + rotate_half(q_rot) * sin_rot
        k_rot_embed = k_rot * cos_rot + rotate_half(k_rot) * sin_rot
        q_embed = jnp.concatenate([q_rot_embed, q_pass], axis=-1)
        k_embed = jnp.concatenate([k_rot_embed, k_pass], axis=-1)
    else:
        # Standard: rotate full head_dim
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
    attention_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Compute multimodal RoPE indices for text + image inputs (video unsupported)."""

    image_token_id = 151655
    vision_start_token_id = 151652

    if input_ids is not None:
        batch, seq_len = input_ids.shape
    else:
        batch = attention_mask.shape[0] if attention_mask is not None else 1
        seq_len = attention_mask.shape[1] if attention_mask is not None else 1

    if input_ids is not None and image_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = jnp.ones_like(total_input_ids)

        position_ids = jnp.ones((3, batch, seq_len), dtype=total_input_ids.dtype)
        mrope_position_deltas: list[jax.Array] = []
        image_index_global = 0

        for i in range(batch):
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
            input_tokens = valid.tolist()

            llm_pos_ids_list: list[jax.Array] = []
            st = 0
            image_index_local = 0

            for _ in range(image_nums):
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1

                if hasattr(image_grid_thw, "ndim") and int(image_grid_thw.ndim) == 3:
                    t, h, w = [int(x) for x in image_grid_thw[i, image_index_local]]
                    image_index_local += 1
                else:
                    t, h, w = [int(x) for x in image_grid_thw[image_index_global]]
                    image_index_global += 1

                llm_grid_t = t
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                text_len = ed_image - st

                if llm_pos_ids_list:
                    st_idx = int(jnp.max(llm_pos_ids_list[-1])) + 1
                else:
                    st_idx = 0

                text_range = jnp.arange(text_len, dtype=valid.dtype)
                text_positions = jnp.tile(text_range.reshape(1, -1), (3, 1)) + st_idx
                llm_pos_ids_list.append(text_positions)

                range_tensor = jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1)
                expanded_range = jnp.tile(range_tensor, (1, llm_grid_h * llm_grid_w))
                t_index = expanded_range.reshape(-1)

                h_index = jnp.arange(llm_grid_h, dtype=jnp.int32).reshape(1, -1, 1)
                h_index = jnp.tile(h_index, (llm_grid_t, 1, llm_grid_w)).reshape(-1)
                w_index = jnp.arange(llm_grid_w, dtype=jnp.int32).reshape(1, 1, -1)
                w_index = jnp.tile(w_index, (llm_grid_t, llm_grid_h, 1)).reshape(-1)
                spatial = jnp.stack([t_index, h_index, w_index], axis=0)
                spatial = spatial + text_len + st_idx
                llm_pos_ids_list.append(spatial)
                st = ed_image + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = int(jnp.max(llm_pos_ids_list[-1])) + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                text_range = jnp.arange(text_len, dtype=valid.dtype)
                text_positions = jnp.tile(text_range.reshape(1, -1), (3, 1)) + st_idx
                llm_pos_ids_list.append(text_positions)

            if llm_pos_ids_list:
                llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            else:
                text_range = jnp.arange(valid.shape[0], dtype=valid.dtype)
                llm_positions = jnp.tile(text_range.reshape(1, -1), (3, 1))

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
    mrope_interleaved: bool
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
    fullatt_block_indexes: Sequence[int]
    num_position_embeddings: Optional[int] = None
    deepstack_visual_indexes: Sequence[int] = ()


@dataclass
class Qwen3VLSpec:
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
    eps: float = 1e-6
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
        return_head_max: bool = False,
    ) -> tuple[jax.Array, Optional["KVCache"], Optional[jax.Array]]:
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

        q = RMSNorm(self.head_dim, self.eps, self.dtype, name="q_norm")(q)
        k = RMSNorm(self.head_dim, self.eps, self.dtype, name="k_norm")(k)

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
            
            # Compute attention scores using batched matmul on flattened (B*Hkv*G) to
            # avoid multi-dimension contractions that METAL struggles to legalize.
            qg = q_grouped.astype(jnp.float32)  # [B, Hkv, G, Q, D]
            kf = k.astype(jnp.float32)          # [B, Hkv, K, D]
            B, Hkv, G, Q, D = qg.shape
            K = kf.shape[2]
            q2 = qg.reshape(B * Hkv * G, Q, D)
            k2 = kf.reshape(B * Hkv, K, D)
            # Repeat keys across the grouped-repeat dimension
            k2 = jnp.repeat(k2, repeats=G, axis=0)  # [B*Hkv*G, K, D]
            # scores: [B*Hkv*G, Q, K]
            attn_scores = jax.lax.dot_general(
                q2,
                jnp.swapaxes(k2, -1, -2),
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.HIGHEST,
            ) * (self.head_dim ** -0.5)
            # Reshape back to (B, H, Q, K)
            attn_scores = attn_scores.reshape(B, Hkv, G, Q, K).reshape(batch, self.num_heads, q.shape[2], k.shape[2])
        else:
            # Non-grouped case: use batched matmul over (B*H)
            qf = q.astype(jnp.float32)  # [B, H, Q, D]
            kf = k.astype(jnp.float32)  # [B, H, K, D]
            B, H, Q, D = qf.shape
            K = kf.shape[2]
            q2 = qf.reshape(B * H, Q, D)
            k2 = kf.reshape(B * H, K, D)
            attn_scores = jax.lax.dot_general(
                q2,
                jnp.swapaxes(k2, -1, -2),
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.HIGHEST,
            ) * (self.head_dim ** -0.5)
            attn_scores = attn_scores.reshape(B, H, Q, K)
        attn_scores = attn_scores + (1.0 - history_mask)[:, None, None, :].astype(jnp.float32) * -1e9  # mask pad
        q_len = attn_scores.shape[2]
        if q_len > 1:
            k_len = attn_scores.shape[3]
            causal = jnp.tril(jnp.ones((q_len, k_len), dtype=jnp.float32))  # causal triangle
            attn_scores = attn_scores + (1.0 - causal)[None, None, :, :] * -1e9
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        head_max: Optional[jax.Array] = None
        if return_head_max:
            # Per-head max attention on the last query position
            head_max = jnp.max(attn_weights[:, :, -1, :], axis=-1)  # [B, H]
        
        # Apply attention weights with grouped-query pattern if needed
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            # Reshape weights for grouped computation
            AWg = attn_weights.reshape(batch, self.num_kv_heads, repeats, q_len, -1)  # [B,Hkv,G,Q,K]
            vf = v.astype(jnp.float32)  # [B, Hkv, K, D]
            B, Hkv, G, Q, K = AWg.shape
            D = vf.shape[-1]
            AW2 = AWg.reshape(B * Hkv * G, Q, K)
            v2 = vf.reshape(B * Hkv, K, D)
            v2 = jnp.repeat(v2, repeats=G, axis=0)  # [B*Hkv*G, K, D]
            # out2: [B*Hkv*G, Q, D]
            out2 = jax.lax.dot_general(
                AW2,
                v2,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.HIGHEST,
            )
            # Reshape back to (B, H, Q, D)
            attn_output = out2.reshape(B, Hkv, G, Q, D).reshape(batch, self.num_heads, q_len, self.head_dim).astype(self.dtype)
        else:
            # Non-grouped case: (B*H) batched matmul
            AW = attn_weights  # [B, H, Q, K]
            vf = v.astype(jnp.float32)  # [B, H, K, D]
            B, H, Q, K = AW.shape
            D = vf.shape[-1]
            AW2 = AW.reshape(B * H, Q, K)
            v2 = vf.reshape(B * H, K, D)
            out2 = jax.lax.dot_general(
                AW2,
                v2,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.HIGHEST,
            )
            attn_output = out2.reshape(B, H, Q, D).astype(self.dtype)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3)).reshape(batch, seqlen, -1)
        out = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype, name="o_proj")(
            attn_output
        )
        return out, cache, head_max


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
            eps=self.eps,
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
        collect_attn: bool = False,
    ) -> tuple[jax.Array, Optional["KVCache"], Optional[jax.Array]]:
        attn_out, cache, head_max = self.attn(
            self.input_norm(hidden_states),
            cos,
            sin,
            mask,
            cache,
            layer_id,
            update_lengths=update_lengths,
            return_head_max=collect_attn,
        )
        hidden_states = hidden_states + attn_out  # res
        hidden_states = hidden_states + self.mlp(self.post_norm(hidden_states))  # res
        return hidden_states, cache, head_max


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


class Qwen3VLModel(nn.Module):
    spec: Qwen3VLSpec
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
            from .vision import Qwen3VisionTransformer

            self.visual = Qwen3VisionTransformer(self.spec.vision)  # optional vision tower
        else:
            self.visual = None

    @staticmethod
    def _apply_deepstack_features(
        hidden: jax.Array, visual_mask: Optional[jax.Array], features: jax.Array
    ) -> jax.Array:
        if visual_mask is None or features.size == 0:
            return hidden

        def _add(batch_hidden, mask_row, feat_row):
            if feat_row.shape[0] == 0:
                return batch_hidden
            idx = jnp.where(mask_row, size=feat_row.shape[0], fill_value=-1)[0]
            valid = idx >= 0
            idx = jnp.where(valid, idx, 0)
            updates = jnp.where(
                valid[:, None],
                feat_row.astype(batch_hidden.dtype),
                jnp.zeros_like(feat_row, dtype=batch_hidden.dtype),
            )
            return batch_hidden.at[idx].add(updates)

        return jax.vmap(_add)(hidden, visual_mask.astype(bool), features)

    def _decode_from_hidden(
        self,
        hidden: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
        visual_mask: Optional[jax.Array] = None,
        deepstack: Optional[tuple[jax.Array, ...]] = None,
        *,
        collect_attn: bool = False,
    ) -> tuple[jax.Array, Optional[KVCache], Optional[jax.Array]]:
        new_cache = cache
        last_layer_idx = len(self.layers) - 1
        deepstack = deepstack or ()
        attn_rows: list[jax.Array] = []
        for layer_id, layer in enumerate(self.layers):
            hidden, new_cache, head_max = layer(
                hidden,
                cos,
                sin,
                mask,
                new_cache,
                layer_id,
                update_lengths=bool(cache is not None and layer_id == last_layer_idx),  # bump cache len at last layer
                collect_attn=collect_attn,
            )
            if collect_attn and head_max is not None:
                attn_rows.append(head_max)  # [B, H]
            if deepstack and layer_id < len(deepstack) and visual_mask is not None:
                hidden = self._apply_deepstack_features(hidden, visual_mask, deepstack[layer_id])
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden.astype(jnp.float32))  # keep logits in fp32
        attn_summary: Optional[jax.Array] = None
        if collect_attn and attn_rows:
            # Stack per-layer rows -> [L, B, H], take last 6 layers -> [N<=6, B, H]
            rows = jnp.stack(attn_rows, axis=0)
            attn_summary = rows[-6:, ...]
        return logits, new_cache, attn_summary

    def forward_text(
        self,
        tokens: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
        *,
        collect_attn: bool = False,
    ) -> tuple[jax.Array, Optional[KVCache], Optional[jax.Array]]:
        hidden = self.embed(tokens)
        return self._decode_from_hidden(
            hidden,
            cos,
            sin,
            mask,
            cache,
            collect_attn=collect_attn,
        )

    def forward_vlm(
        self,
        tokens: jax.Array,
        vision_embeds: Union[jax.Array, VisionEmbeddings],
        image_pad_id: int,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
        *,
        collect_attn: bool = False,
    ) -> tuple[jax.Array, Optional[KVCache], Optional[jax.Array]]:
        hidden = self.embed(tokens)
        batch = hidden.shape[0]

        if isinstance(vision_embeds, VisionEmbeddings):
            vision_pack = vision_embeds.cast(self.dtype).with_batch_dim(batch)
        else:
            vision_arr = jnp.asarray(vision_embeds, dtype=self.dtype)
            if vision_arr.ndim == 2:
                vision_arr = vision_arr[None, ...]
            if vision_arr.shape[0] not in (1, batch):
                raise ValueError(
                    "vision_embeds batch dimension must be 1 or match tokens batch"
                )
            if vision_arr.shape[0] == 1 and batch > 1:
                vision_arr = jnp.tile(vision_arr, (batch, 1, 1))
            vision_pack = VisionEmbeddings(tokens=vision_arr, deepstack=())

        visual_mask = (tokens == jnp.int32(image_pad_id))

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

        hidden = jax.vmap(_inject)(hidden, tokens, vision_pack.tokens)
        return self._decode_from_hidden(
            hidden,
            cos,
            sin,
            mask,
            cache,
            visual_mask=visual_mask,
            deepstack=vision_pack.deepstack,
            collect_attn=collect_attn,
        )

    def forward_vlm_hidden(
        self,
        tokens: jax.Array,
        vision_embeds: Union[jax.Array, VisionEmbeddings],
        image_pad_id: int,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[jax.Array, Optional[KVCache]]:
        """Same as forward_vlm but returns the final normalized hidden states.

        This is useful for memory-lean losses that avoid materializing the
        full [B, T, V] logits tensor. The returned hidden is the output of
        the final RMSNorm and matches the input expected by `lm_head` when
        cast to fp32.
        """
        hidden = self.embed(tokens)
        batch = hidden.shape[0]

        if isinstance(vision_embeds, VisionEmbeddings):
            vision_pack = vision_embeds.cast(self.dtype).with_batch_dim(batch)
        else:
            vision_arr = jnp.asarray(vision_embeds, dtype=self.dtype)
            if vision_arr.ndim == 2:
                vision_arr = vision_arr[None, ...]
            if vision_arr.shape[0] not in (1, batch):
                raise ValueError(
                    "vision_embeds batch dimension must be 1 or match tokens batch"
                )
            if vision_arr.shape[0] == 1 and batch > 1:
                vision_arr = jnp.tile(vision_arr, (batch, 1, 1))
            vision_pack = VisionEmbeddings(tokens=vision_arr, deepstack=())

        visual_mask = (tokens == jnp.int32(image_pad_id))

        def _inject(batch_hidden, batch_tokens, batch_vis):
            num_vision = batch_vis.shape[0]
            pad_positions = jnp.where(
                batch_tokens == jnp.int32(image_pad_id), size=num_vision, fill_value=-1
            )[0]
            valid = pad_positions >= 0
            pad_positions = jnp.where(valid, pad_positions, 0)
            updates = jnp.where(valid[:, None], batch_vis.astype(batch_hidden.dtype), batch_hidden[pad_positions])
            batch_hidden = batch_hidden.at[pad_positions].set(updates)
            return batch_hidden

        hidden = jax.vmap(_inject)(hidden, tokens, vision_pack.tokens)

        new_cache = cache
        last_layer_idx = len(self.layers) - 1
        deepstack = vision_pack.deepstack or ()
        for layer_id, layer in enumerate(self.layers):
            hidden, new_cache, _ = layer(
                hidden,
                cos,
                sin,
                mask,
                new_cache,
                layer_id,
                update_lengths=bool(cache is not None and layer_id == last_layer_idx),
            )
            if deepstack and layer_id < len(deepstack) and visual_mask is not None:
                hidden = self._apply_deepstack_features(hidden, visual_mask, deepstack[layer_id])

        hidden = self.final_norm(hidden)
        return hidden, new_cache

    def encode_vision(self, pixel_values: jax.Array, grid_thw: jax.Array) -> VisionEmbeddings:
        if self.visual is None:
            raise ValueError("Vision backbone not configured for this model")
        tokens, deepstack = self.visual(pixel_values, grid_thw)
        return VisionEmbeddings(tokens=tokens, deepstack=tuple(deepstack))

    def decode_step(
        self,
        token: jax.Array,
        cache: KVCache,
        rope_deltas: Optional[jax.Array],
        mask: Optional[jax.Array] = None,
        *,
        collect_attn: bool = False,
    ) -> tuple[jax.Array, KVCache, jax.Array]:
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
        logits, new_cache, attn_summary = self.forward_text(
            step_tokens,
            cos,
            sin,
            mask=mask,
            cache=cache,
            collect_attn=collect_attn,
        )
        # Normalize attn_summary to [N, H] (batch squeezed) with stable shape when disabled
        if attn_summary is None:
            attn_out = jnp.zeros((6, int(self.spec.text.num_attention_heads)), dtype=jnp.float32)
        else:
            attn_out = attn_summary[:, 0, :].astype(jnp.float32)
        return logits[:, -1, :], new_cache, attn_out

    def __call__(
        self,
        tokens: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[jax.Array, Optional[KVCache]]:
        logits, cache = self.forward_text(tokens, cos, sin, mask=mask, cache=cache)[:2]
        return logits, cache


def _load_hf_config(hf_dir: str) -> dict[str, Any]:
    with open(f"{hf_dir}/config.json") as f:
        return json.load(f)


def spec_from_config(cfg: dict[str, Any]) -> Qwen3VLSpec:
    # Map HF config to our internal spec
    text_cfg = cfg.get("text_config", cfg)
    rope_cfg = text_cfg.get("rope_scaling", cfg.get("rope_scaling", {}))
    # Compute common head_dim once; needed for RoPE section validation
    _head_dim = text_cfg.get("head_dim")
    if _head_dim is None:
        _head_dim = text_cfg["hidden_size"] // text_cfg["num_attention_heads"]
    else:
        _head_dim = int(_head_dim)
    vision_cfg = cfg.get("vision_config")

    # Parse optional rope scaling type/factor
    rope_scaling_type = None
    rope_scaling_factor = None
    rope_interleaved = False
    if isinstance(rope_cfg, dict):
        rope_scaling_type = rope_cfg.get("type")
        rope_scaling_factor = rope_cfg.get("factor", rope_cfg.get("finetuned_factor"))
        rope_interleaved = bool(rope_cfg.get("mrope_interleaved", False))

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
        total = sum(rope_section)
        if not rope_interleaved and total != expected:
            raise ValueError(
                f"Sum of rope_scaling.mrope_section must equal head_dim//2 ({expected}); got {total}"
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
        mrope_interleaved=rope_interleaved,
        rms_norm_eps=text_cfg["rms_norm_eps"],
        vocab_size=text_cfg["vocab_size"],
    )
    vision = None
    if vision_cfg is not None:
        patch_sz = vision_cfg.get("patch_size", vision_cfg.get("spatial_patch_size"))
        temporal_patch_sz = vision_cfg.get("temporal_patch_size", vision_cfg.get("temporal_patch", 1))
        window_size = vision_cfg.get(
            "window_size",
            (patch_sz or 1) * vision_cfg.get("spatial_merge_size", 1),
        )
        vision = VisionBackboneSpec(
            hidden_size=vision_cfg["hidden_size"],
            out_hidden_size=vision_cfg["out_hidden_size"],
            depth=vision_cfg["depth"],
            num_heads=vision_cfg["num_heads"],
            intermediate_size=vision_cfg["intermediate_size"],
            patch_size=patch_sz,
            temporal_patch_size=temporal_patch_sz,
            spatial_merge_size=vision_cfg["spatial_merge_size"],
            window_size=window_size,
            in_channels=vision_cfg.get("in_channels", vision_cfg.get("in_chans", 3)),
            fullatt_block_indexes=vision_cfg.get(
                "fullatt_block_indexes",
                vision_cfg.get("deepstack_visual_indexes", []),
            ),
            num_position_embeddings=vision_cfg.get("num_position_embeddings"),
            deepstack_visual_indexes=tuple(vision_cfg.get("deepstack_visual_indexes", [])),
        )
    return Qwen3VLSpec(
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
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"layers_\1/attn/q_norm/weight",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"layers_\1/attn/q_norm/weight",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers_\1/attn/k_proj/kernel",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers_\1/attn/k_proj/kernel",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": r"layers_\1/attn/k_proj/bias",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": r"layers_\1/attn/k_proj/bias",
    r"model\.language_model\.model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"layers_\1/attn/k_norm/weight",
    r"model\.language_model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"layers_\1/attn/k_norm/weight",
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
    r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"layers_\1/attn/q_norm/weight",
    r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers_\1/attn/k_proj/kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": r"layers_\1/attn/k_proj/bias",
    r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"layers_\1/attn/k_norm/weight",
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
    r"model\.visual\.deepstack_merger_list\.([0-9]+)\.norm\.weight": r"visual/deepstack_mergers_\1/norm/weight",
    r"visual\.deepstack_merger_list\.([0-9]+)\.norm\.weight": r"visual/deepstack_mergers_\1/norm/weight",
    r"model\.visual\.deepstack_merger_list\.([0-9]+)\.linear_fc1\.weight": r"visual/deepstack_mergers_\1/linear1/kernel",
    r"visual\.deepstack_merger_list\.([0-9]+)\.linear_fc1\.weight": r"visual/deepstack_mergers_\1/linear1/kernel",
    r"model\.visual\.deepstack_merger_list\.([0-9]+)\.linear_fc1\.bias": r"visual/deepstack_mergers_\1/linear1/bias",
    r"visual\.deepstack_merger_list\.([0-9]+)\.linear_fc1\.bias": r"visual/deepstack_mergers_\1/linear1/bias",
    r"model\.visual\.deepstack_merger_list\.([0-9]+)\.linear_fc2\.weight": r"visual/deepstack_mergers_\1/linear2/kernel",
    r"visual\.deepstack_merger_list\.([0-9]+)\.linear_fc2\.weight": r"visual/deepstack_mergers_\1/linear2/kernel",
    r"model\.visual\.deepstack_merger_list\.([0-9]+)\.linear_fc2\.bias": r"visual/deepstack_mergers_\1/linear2/bias",
    r"visual\.deepstack_merger_list\.([0-9]+)\.linear_fc2\.bias": r"visual/deepstack_mergers_\1/linear2/bias",
}


def _torch_key_to_flax(key: str) -> Optional[str]:
    for mapping in (_TEXT_KEY_RULES, _VISION_KEY_RULES):
        for pattern, target in mapping.items():
            if re.match(pattern, key):
                return re.sub(pattern, target, key)
    return None


def create_model_from_hf(hf_dir: str, dtype: Optional[str] = None) -> tuple[Qwen3VLModel, dict[str, Any]]:
    cfg = _load_hf_config(hf_dir)
    spec = spec_from_config(cfg)
    if dtype is not None:
        dtype_map = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16, "fp32": jnp.float32}
        model = Qwen3VLModel(spec, dtype=dtype_map.get(str(dtype).lower(), jnp.bfloat16))
    else:
        model = Qwen3VLModel(spec)

    dummy_ids = jnp.zeros((1, 1), dtype=jnp.int32)
    rope_axes = len(tuple(int(x) for x in spec.text.rope_section))
    rope_dim = sum(int(x) for x in spec.text.rope_section) * 2
    dummy_cos = jnp.zeros((rope_axes, 1, 1, rope_dim), dtype=model.dtype)
    dummy_sin = jnp.zeros((rope_axes, 1, 1, rope_dim), dtype=model.dtype)
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

    # Handle tied embeddings: copy embed weights to lm_head when tie_word_embeddings is True
    if cfg.get("tie_word_embeddings", False):
        if "embed" in param_dict and "embedding" in param_dict["embed"]:
            # lm_head kernel should be the transpose of the embedding
            # embedding: (vocab_size, hidden_size)
            # lm_head kernel: (hidden_size, vocab_size)
            embed_weight = param_dict["embed"]["embedding"]
            param_dict.setdefault("lm_head", {})["kernel"] = embed_weight.T
            print(f"Note: tie_word_embeddings is True, copying embed weights to lm_head (transposed)")

    return model, flax.core.freeze(param_dict)


def create_model_from_ckpt(ckpt_dir: str, dtype: Optional[str] = None) -> tuple[Qwen3VLModel, dict[str, Any]]:
    from vlmrl.utils.checkpoint import Checkpoint

    cfg = _load_hf_config(ckpt_dir)
    spec = spec_from_config(cfg)
    if dtype is not None:
        dtype_map = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16, "fp32": jnp.float32}
        model = Qwen3VLModel(spec, dtype=dtype_map.get(str(dtype).lower(), jnp.bfloat16))
    else:
        model = Qwen3VLModel(spec)
    ckpt = Checkpoint(f"{ckpt_dir}/params.pkl", parallel=False)
    params = ckpt.load_as_dict()["params"]
    return model, params


__all__ = [
    "Qwen3VLModel",
    "KVCache",
    "apply_multimodal_rotary_pos_emb",
    "build_text_rope",
    "build_mrope",
    "get_rope_index",
    "spec_from_config",
    "create_model_from_hf",
    "create_model_from_ckpt",
]
