"""Lightweight LoRA utilities for Flax models.

This module provides:
- LoRAConfig: which layers to adapt and basic hyperparameters.
- LoRADense: a drop-in Dense that adds a low-rank adapter (A, B) on top of the base kernel.

Design choices
- Keep base kernel param names identical to nn.Dense ("kernel"/"bias") for easy checkpoint loading.
- Add LoRA params as "lora_A" and "lora_B" so they are easy to mask in the optimizer.
- Use standard scaling alpha/r and A uniform init ~ U(-1/d_in, 1/d_in), B zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


DType = jnp.dtype


@dataclass
class LoRAConfig:
    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    apply_attn: bool = True
    apply_mlp: bool = True
    apply_vision: bool = False


class LoRADense(nn.Module):
    """Dense with optional LoRA adapter.

    When enabled, output = x @ W + b + (alpha/r) * (x @ A) @ B,
    with A in R^{in,r}, B in R^{r,out}. Base kernel/bias names match nn.Dense for compatibility.
    """

    features: int
    use_bias: bool = False
    dtype: DType = jnp.bfloat16
    lora_rank: int = 0
    lora_alpha: int = 32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Keep a float32 copy for LoRA path stability; use module dtype for base path.
        x32 = x.astype(jnp.float32)
        x = x.astype(self.dtype)
        in_features = x.shape[-1]

        # Base Dense kernel/bias (match nn.Dense param names)
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (in_features, self.features),
            self.dtype,
        )
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.dtype)
        else:
            bias = None

        y = jnp.matmul(x, kernel)
        if bias is not None:
            y = y + bias

        # LoRA adapter
        r = int(self.lora_rank or 0)
        if r > 0:
            # A: (in, r) uniform ~ U(-1/d_in, 1/d_in) in float32 for stability
            scale = 1.0 / float(in_features if in_features > 0 else 1)
            def _init_A(key, shape, dtype):
                return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

            A = self.param("lora_A", _init_A, (in_features, r), jnp.float32)
            B = self.param("lora_B", nn.initializers.zeros, (r, self.features), jnp.float32)
            lora_out32 = jnp.matmul(jnp.matmul(x32, A), B)
            lora_out32 = (float(self.lora_alpha) / float(r)) * lora_out32
            y = y + lora_out32.astype(self.dtype)

        return y.astype(self.dtype)


__all__ = ["LoRAConfig", "LoRADense"]
