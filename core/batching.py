"""Shared padding and masking helpers (NumPy host, JAX device).

This module copies the minimal padding/masking logic used in the current
training loops into a reusable place, keeping dtype and shape semantics intact.
It does not modify any existing modules.

Notes
- Padding for text uses NumPy on host and converts to jnp on return for
  predictable memory behavior.
- Vision padding preserves original dtypes and supports VisionEmbeddings.
- Masks target text targets (tokens[:, 1:]), not the original sequence length.
"""

from __future__ import annotations

import os
from typing import List, Union

import jax
import jax.numpy as jnp
import numpy as np

from vlmrl.models.qwen3vl.model import VisionEmbeddings


_DEBUG_PAD = os.environ.get("VLMRL_DEBUG_PAD", "") not in ("", "0", "false", "False", "FALSE")
_SKIP_DEEPSTACK = os.environ.get("VLMRL_SKIP_DEEPSTACK", "") not in ("", "0", "false", "False", "FALSE")
_PAD_VISION_LOGGED = False  # ensure we only spam once per process


def set_skip_deepstack(flag: bool) -> None:
    """Toggle whether deepstack features are returned by pad_vision.

    When enabled, pad_vision will return VisionEmbeddings with an empty
    deepstack tuple to reduce host/device memory usage. This does not modify
    vision encoding itself; it only avoids staging deep features into the
    training batch.
    """
    global _SKIP_DEEPSTACK
    _SKIP_DEEPSTACK = bool(flag)


def _maybe_print_debug(msg: str, *, prefix: str = "pad_vision") -> None:
    # Emit only when enabled and from process 0 to avoid duplicate logs
    if _DEBUG_PAD and jax.process_index() == 0:
        print(f"[{prefix}] {msg}")


def pad_sequences(seqs: List[np.ndarray], pad_val: int, max_len: int | None = None) -> jnp.ndarray:
    """Pad a list of integer sequences into a dense [B, T] array.

    Args:
        seqs: variable-length arrays shaped [T_i]
        pad_val: integer pad token id
        max_len: optional cap on the padded length
    Returns:
        jnp.ndarray[int32] of shape [B, T]
    """
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


def pad_float_sequences(
    seqs: List[np.ndarray],
    pad_val: float = 0.0,
    max_len: int | None = None,
) -> jnp.ndarray:
    """Pad a list of float sequences into a dense [B, T] array."""
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


def pad_vision(arrs: List[Union[np.ndarray, VisionEmbeddings]]) -> Union[jnp.ndarray, VisionEmbeddings]:
    """Pad a list of vision embeddings into a batch.

    Supports both raw arrays shaped [tokens, dim] and VisionEmbeddings.
    For VisionEmbeddings, preserves dtype and optionally drops deepstack when
    VLMRL_SKIP_DEEPSTACK=1 (or toggled via set_skip_deepstack).
    """
    if not arrs:
        return jnp.zeros((0, 0, 0), dtype=jnp.float32)

    first = arrs[0]
    if isinstance(first, VisionEmbeddings):
        dim = int(first.tokens.shape[-1]) if first.tokens.ndim == 2 else int(first.tokens.shape[-1])
        max_tokens = max(int(elem.tokens.shape[0]) for elem in arrs)
        # Preserve original dtype to avoid unnecessary upcasting and copies
        tokens_dtype = np.asarray(first.tokens).dtype
        tokens_out = np.zeros((len(arrs), max_tokens, dim), dtype=tokens_dtype)
        deepstack_levels = 0 if _SKIP_DEEPSTACK else len(first.deepstack)
        deepstack_dtypes = [np.asarray(first.deepstack[i]).dtype for i in range(deepstack_levels)]
        deepstack_out = (
            [np.zeros((len(arrs), max_tokens, dim), dtype=deepstack_dtypes[i]) for i in range(deepstack_levels)]
            if deepstack_levels > 0 else []
        )

        # One-time debug summary on process 0
        global _PAD_VISION_LOGGED
        if not _PAD_VISION_LOGGED:
            counts = [int(elem.tokens.shape[0]) for elem in arrs]
            _maybe_print_debug(
                (
                    f"vision batch count={len(arrs)}, max_tokens={max_tokens}, dim={dim}, "
                    f"deepstack_levels={deepstack_levels}, tokens_dtype={tokens_dtype}"
                )
            )
            _maybe_print_debug(
                f"per-sample token counts: min={min(counts)}, max={max(counts)}, mean={float(np.mean(counts)):.1f}"
            )
            if deepstack_levels > 0:
                _maybe_print_debug(
                    "deepstack dtypes=" + ", ".join(str(dt) for dt in deepstack_dtypes)
                )
            if _SKIP_DEEPSTACK:
                _maybe_print_debug("deepstack copying skipped via VLMRL_SKIP_DEEPSTACK=1")
            _PAD_VISION_LOGGED = True

        for i, emb in enumerate(arrs):
            tokens = np.asarray(emb.tokens, dtype=tokens_dtype)
            tokens_out[i, :tokens.shape[0], :] = tokens
            if deepstack_levels > 0 and len(emb.deepstack) != deepstack_levels:
                raise ValueError("Inconsistent DeepStack depth across vision embeddings")
            if deepstack_levels > 0:
                for level_idx, feat in enumerate(emb.deepstack):
                    feat_np = np.asarray(feat, dtype=deepstack_dtypes[level_idx])
                    deepstack_out[level_idx][i, :feat_np.shape[0], :] = feat_np

        return VisionEmbeddings(
            tokens=jnp.asarray(tokens_out),
            deepstack=tuple(jnp.asarray(layer) for layer in deepstack_out) if deepstack_out else (),
        )

    # Raw array path
    dim = int(first.shape[-1])
    max_tokens = max(int(arr.shape[0]) for arr in arrs)
    base_dtype = np.asarray(first).dtype
    out = np.zeros((len(arrs), max_tokens, dim), dtype=base_dtype)
    _maybe_print_debug(
        f"vision batch count={len(arrs)}, max_tokens={max_tokens}, dim={dim}, deepstack_levels=0, dtype={base_dtype}"
    )
    for i, arr in enumerate(arrs):
        out[i, :arr.shape[0], :] = np.asarray(arr, dtype=base_dtype)
    return jnp.asarray(out)


def pad_grid(arrs: List[np.ndarray]) -> jnp.ndarray:
    """Pad a list of THW grids into a dense [B, N, 3] array."""
    if not arrs:
        return jnp.zeros((0, 0, 3), dtype=jnp.int32)
    max_rows = max(int(arr.shape[0]) for arr in arrs)
    out = np.zeros((len(arrs), max_rows, 3), dtype=np.int32)
    for i, arr in enumerate(arrs):
        np_arr = np.asarray(arr, dtype=np.int32)
        out[i, :np_arr.shape[0], :] = np_arr
    return jnp.asarray(out)


def build_masks(prompt_lens: np.ndarray, action_lens: np.ndarray, seq_len: int) -> jnp.ndarray:
    """Build target-side masks for policy gradient over tokens[:, 1:].

    Returns a [B, T-1] mask that selects only action tokens following the
    prompt prefix. This mirrors current training semantics where the model is
    trained against text_target = tokens[:, 1:].
    """
    if seq_len <= 1:
        return jnp.zeros((len(prompt_lens), 0), dtype=jnp.int32)
    mask = np.zeros((len(prompt_lens), seq_len - 1), dtype=np.int32)
    for i, (prompt_len, action_len) in enumerate(zip(prompt_lens, action_lens, strict=True)):
        action_len = int(action_len)
        if action_len <= 0:
            continue
        prompt_len = int(prompt_len)
        start = max(prompt_len - 1, 0)
        end = min(start + action_len - 1, mask.shape[1] - 1)
        if start <= end:
            mask[i, start : end + 1] = 1
    return jnp.asarray(mask)


__all__ = [
    "set_skip_deepstack",
    "pad_sequences",
    "pad_float_sequences",
    "pad_vision",
    "pad_grid",
    "build_masks",
]

