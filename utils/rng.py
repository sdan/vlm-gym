"""Lightweight RNG sequence helper for JAX.

Avoids manual splitting clutter by providing a simple `.next()` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax


@dataclass
class RngSeq:
    seed: int

    def __post_init__(self):
        self._key = jax.random.PRNGKey(int(self.seed))

    def next(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub


__all__ = ["RngSeq"]

