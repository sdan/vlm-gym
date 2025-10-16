"""Vision backbone for Qwen3-VL."""

from __future__ import annotations

from typing import List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .model import rotate_half


from .model import VisionBackboneSpec


DType = jnp.dtype


class VisionRotaryEmbedding(nn.Module):
    dim: int
    theta: float = 10000.0

    def __call__(self, seq_len: int) -> jax.Array:
        # 1D rope table for ViT
        # Use half-dim directly: produce self.dim distinct frequencies.
        # This pairs with a later duplication to reach full head_dim.
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, dtype=jnp.float32) / self.dim))
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        return jnp.outer(positions, inv_freq)


class VisionPatchEmbed(nn.Module):
    embed_dim: int
    patch_volume: int
    dtype: DType = jnp.bfloat16

    @nn.compact
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        # linear patch projection
        proj = nn.Dense(self.embed_dim, use_bias=False, dtype=self.dtype, name="proj")
        return proj(hidden_states.astype(self.dtype))


class VisionAttention(nn.Module):
    hidden_size: int
    num_heads: int
    dtype: DType = jnp.bfloat16

    def setup(self) -> None:
        self.head_dim = self.hidden_size // self.num_heads  # per-head width
        self.scale = self.head_dim ** -0.5

    @nn.compact
    def __call__(
        self,
        hidden_states: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        cu_seqlens: jax.Array,
    ) -> jax.Array:
        qkv = nn.Dense(3 * self.hidden_size, use_bias=True, dtype=self.dtype, name="qkv")(hidden_states)  # fused
        q, k, v = jnp.split(qkv, 3, axis=-1)

        seq_len = hidden_states.shape[0]
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_heads, self.head_dim)

        # cos/sin have shape (seq_len, 2*head_dim) due to duplication
        # We need to take only the first head_dim elements to match q/k shape
        cos = cos.astype(self.dtype)[:, :self.head_dim]
        sin = sin.astype(self.dtype)[:, :self.head_dim]
        # Expand for broadcasting with (seq_len, num_heads, head_dim)
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin

        num_windows = cu_seqlens.shape[0] - 1
        attn_chunks: List[jax.Array] = []

        for i in range(num_windows):
            start = int(cu_seqlens[i])
            end = int(cu_seqlens[i + 1])
            if start >= end:
                continue

            q_chunk = q_embed[start:end]
            k_chunk = k_embed[start:end]
            v_chunk = v[start:end]

            q_chunk = jnp.transpose(q_chunk, (1, 0, 2))
            k_chunk = jnp.transpose(k_chunk, (1, 0, 2))
            v_chunk = jnp.transpose(v_chunk, (1, 0, 2))

            attn_scores = jnp.einsum(
                "hqd,hkd->hqk",
                q_chunk.astype(jnp.float32),
                k_chunk.astype(jnp.float32),
            ) * self.scale
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            attn_out = jnp.einsum(
                "hqk,hkd->hqd",
                attn_weights,
                v_chunk.astype(jnp.float32),
            ).astype(self.dtype)

            attn_out = jnp.transpose(attn_out, (1, 0, 2))
            attn_chunks.append(attn_out)

        attn_output = jnp.concatenate(attn_chunks, axis=0).reshape(seq_len, self.hidden_size)  # stitch
        return nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="proj")(attn_output)


class VisionBlock(nn.Module):
    spec: "VisionBackboneSpec"
    dtype: DType = jnp.bfloat16

    def setup(self) -> None:
        # Import lazily to avoid circular dependency at module import time.
        from .model import RMSNorm, FeedForward

        self.norm1 = RMSNorm(self.spec.hidden_size, 1e-6, self.dtype)
        self.norm2 = RMSNorm(self.spec.hidden_size, 1e-6, self.dtype)
        self.attn = VisionAttention(self.spec.hidden_size, self.spec.num_heads, self.dtype)
        self.mlp = FeedForward(
            hidden_size=self.spec.hidden_size,
            intermediate_size=self.spec.intermediate_size,
            dtype=self.dtype,
            use_bias=True,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        cu_seqlens: jax.Array,
    ) -> jax.Array:
        attn_out = self.attn(self.norm1(hidden_states), cos, sin, cu_seqlens)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionPatchMerger(nn.Module):
    context_dim: int
    out_dim: int
    spatial_merge_size: int
    use_postshuffle_norm: bool = False
    dtype: DType = jnp.bfloat16

    def setup(self) -> None:
        from .model import RMSNorm

        self.unit = self.spatial_merge_size ** 2  # tokens per merge
        self.hidden_size = self.context_dim * self.unit
        norm_dim = self.hidden_size if self.use_postshuffle_norm else self.context_dim
        self.norm = RMSNorm(norm_dim, 1e-6, self.dtype)
        self.linear1 = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="linear1")
        self.linear2 = nn.Dense(self.out_dim, use_bias=True, dtype=self.dtype, name="linear2")

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        if hidden_states.shape[0] % self.unit != 0:
            raise ValueError("Vision sequence must align with spatial_merge_size**2")
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.reshape(-1, self.unit * self.context_dim)
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.reshape(-1, self.unit * self.context_dim)
        hidden_states = nn.gelu(self.linear1(hidden_states))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class Qwen3VisionTransformer(nn.Module):
    spec: "VisionBackboneSpec"
    dtype: DType = jnp.float32

    def setup(self) -> None:
        patch_volume = (
            self.spec.in_channels
            * self.spec.temporal_patch_size
            * self.spec.patch_size
            * self.spec.patch_size
        )
        self.patch_embed = VisionPatchEmbed(self.spec.hidden_size, patch_volume, self.dtype)
        rotary_dim = (self.spec.hidden_size // self.spec.num_heads) // 2
        self.rotary = VisionRotaryEmbedding(rotary_dim)
        self.blocks = [VisionBlock(self.spec, dtype=self.dtype) for _ in range(self.spec.depth)]
        self.merger = VisionPatchMerger(
            context_dim=self.spec.hidden_size,
            out_dim=self.spec.out_hidden_size,
            spatial_merge_size=self.spec.spatial_merge_size,
            dtype=self.dtype,
        )
        self.deepstack_visual_indexes = tuple(self.spec.deepstack_visual_indexes)
        self.deepstack_mergers = [
            VisionPatchMerger(
                context_dim=self.spec.hidden_size,
                out_dim=self.spec.out_hidden_size,
                spatial_merge_size=self.spec.spatial_merge_size,
                use_postshuffle_norm=True,
                dtype=self.dtype,
            )
            for _ in self.deepstack_visual_indexes
        ]

    def _rot_pos_emb(self, grid_thw: jax.Array) -> jax.Array:
        grid_thw = jnp.asarray(grid_thw, dtype=jnp.int32)
        pos_chunks: List[jax.Array] = []

        for idx in range(grid_thw.shape[0]):
            t, h, w = grid_thw[idx]
            merge = self.spec.spatial_merge_size
            hpos_ids = jnp.arange(h, dtype=jnp.int32)[:, None].repeat(w, axis=1)
            wpos_ids = jnp.arange(w, dtype=jnp.int32)[None, :].repeat(h, axis=0)
            hpos_ids = hpos_ids.reshape(h // merge, merge, w // merge, merge)
            hpos_ids = hpos_ids.transpose(0, 2, 1, 3).reshape(-1)
            wpos_ids = wpos_ids.reshape(h // merge, merge, w // merge, merge)
            wpos_ids = wpos_ids.transpose(0, 2, 1, 3).reshape(-1)
            pos = jnp.stack([hpos_ids, wpos_ids], axis=-1)
            pos = jnp.tile(pos, (int(t), 1))
            pos_chunks.append(pos)

        pos_ids = jnp.concatenate(pos_chunks, axis=0)
        max_grid_size = int(jnp.max(grid_thw[:, 1:]))
        rotary_pos_emb_full = self.rotary(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        return rotary_pos_emb

    def _get_window_index(self, grid_thw: jax.Array) -> Tuple[jax.Array, jax.Array]:
        grid_thw = jnp.asarray(grid_thw, dtype=jnp.int32)
        window_indices: List[jax.Array] = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window = (
            self.spec.window_size // self.spec.spatial_merge_size // self.spec.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spec.spatial_merge_size
            llm_grid_w = grid_w // self.spec.spatial_merge_size
            index = jnp.arange(grid_t * llm_grid_h * llm_grid_w, dtype=jnp.int32).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = (vit_merger_window - (llm_grid_h % vit_merger_window)) % vit_merger_window  # pad to window
            pad_w = (vit_merger_window - (llm_grid_w % vit_merger_window)) % vit_merger_window
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window

            index_padded = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)  # sentinel
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window,
                num_windows_w,
                vit_merger_window,
            )
            index_padded = index_padded.transpose(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window,
                vit_merger_window,
            )

            seqlens = (index_padded != -100).sum(axis=(2, 3)).reshape(-1)  # valid per-window size
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]

            window_indices.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                jnp.cumsum(seqlens) * (self.spec.spatial_merge_size ** 2) + cu_window_seqlens[-1]
            )  # account for merge unit
            cu_window_seqlens.extend(list(cu_seqlens_tmp))
            window_index_id += int(grid_t * llm_grid_h * llm_grid_w)

        window_index = jnp.concatenate(window_indices, axis=0)
        cu_window_seqlens_arr = jnp.array(cu_window_seqlens, dtype=jnp.int32)
        mask = jnp.concatenate(
            [jnp.array([True], dtype=bool), cu_window_seqlens_arr[1:] != cu_window_seqlens_arr[:-1]]
        )  # drop duplicates
        cu_window_seqlens_arr = cu_window_seqlens_arr[mask]
        return window_index, cu_window_seqlens_arr

    def __call__(self, pixel_values: jax.Array, grid_thw: jax.Array) -> jax.Array:
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb = self._rot_pos_emb(grid_thw)  # per-token sin/cos
        window_index, cu_window_seqlens = self._get_window_index(grid_thw)

        seq_len = hidden_states.shape[0]
        spatial_merge_unit = self.spec.spatial_merge_size ** 2
        hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)

        # Duplicate half-dim angles to match full head_dim (rotate_half expects full dim).
        emb = jnp.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        cos = jnp.cos(emb).astype(self.dtype)
        sin = jnp.sin(emb).astype(self.dtype)

        cu_seqlens_list = []
        spatial_merge_unit = self.spec.spatial_merge_size ** 2
        for grid_t, grid_h, grid_w in grid_thw:
            cu_seqlens_list.append(int(grid_t * grid_h * grid_w * spatial_merge_unit))
        cu_seqlens = jnp.cumsum(jnp.array(cu_seqlens_list, dtype=jnp.int32))
        cu_seqlens = jnp.concatenate([jnp.array([0], dtype=jnp.int32), cu_seqlens])

        fullatt_idxs = set(self.spec.fullatt_block_indexes)
        deepstack_features: List[jax.Array] = []
        for layer_idx, block in enumerate(self.blocks):
            cu_seqlens_now = cu_seqlens if layer_idx in fullatt_idxs else cu_window_seqlens
            hidden_states = block(hidden_states, cos, sin, cu_seqlens_now)
            if layer_idx in self.deepstack_visual_indexes:
                merger_idx = len(deepstack_features)
                if merger_idx >= len(self.deepstack_mergers):
                    raise ValueError(
                        "deepstack_visual_indexes length exceeds configured DeepStack mergers."
                    )
                feature = self.deepstack_mergers[merger_idx](hidden_states)
                deepstack_features.append(feature)

        hidden_states = self.merger(hidden_states)
        reverse_indices = jnp.argsort(window_index)  # undo window shuffle
        hidden_states = hidden_states[reverse_indices, :]
        deepstack_features = [feat[reverse_indices, :] for feat in deepstack_features]
        return hidden_states, tuple(deepstack_features)


__all__ = ["Qwen3VisionTransformer"]
