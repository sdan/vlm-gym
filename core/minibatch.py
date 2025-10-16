"""Minibatch and sharding helpers for PPO-style updates.

This module provides small utilities to slice tensors or structured
embeddings into minibatches, replacing repeated reshape/index math.
"""

from __future__ import annotations

from typing import Any, Tuple

import jax.numpy as jnp

from vlmrl.models.qwen3vl.model import VisionEmbeddings


def _trim_for_minibatch(x, mb: int) -> Tuple[Any, int, int]:
    """Trim the leading dimension to a multiple of `mb`.

    Returns (trimmed, chunks, size_per_chunk) where `chunks = N // mb`.
    """
    if mb <= 0:
        raise ValueError("minibatch size must be positive")
    n = int(x.shape[0])
    chunks = max(1, n // mb)
    n_use = chunks * mb
    return x[:n_use], chunks, mb


def shard_for_minibatch(x: Any, mb: int) -> Any:
    """Reshape the leading dimension into [chunks, mb, ...].

    - For jnp.ndarray shaped [N, ...], returns [chunks, mb, ...].
    - For VisionEmbeddings with tokens [N, L, D] and optional deepstack, shards
      both tokens and deepstack to [chunks, mb, L, D].
    """
    if isinstance(x, VisionEmbeddings):
        tokens_trim, chunks, mb_size = _trim_for_minibatch(x.tokens, mb)
        tokens_sh = tokens_trim.reshape(chunks, mb_size, *tokens_trim.shape[1:])
        if x.deepstack:
            deep_sh = tuple(ds[: chunks * mb_size].reshape(chunks, mb_size, *ds.shape[1:]) for ds in x.deepstack)
        else:
            deep_sh = ()
        return VisionEmbeddings(tokens=tokens_sh, deepstack=deep_sh)

    # Default array path
    x_trim, chunks, mb_size = _trim_for_minibatch(x, mb)
    new_shape = (chunks, mb_size) + tuple(int(d) for d in x.shape[1:])
    return x_trim.reshape(new_shape)


__all__ = ["shard_for_minibatch"]

