"""Typed containers for sampling, batching, and rollouts.

This module introduces small, explicit dataclasses that make shapes and types
clear at call sites, without modifying any existing modules. These types are
intended for the new policy/sampler/ppo paths and can coexist with the current
dictionary-based flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import jax.numpy as jnp

from vlmrl.models.qwen3vl.model import VisionEmbeddings


@dataclass
class SamplingConfig:
    """Lightweight sampling configuration.

    - temperature: float sampling temperature (> 0)
    - top_p: nucleus sampling probability (0 < top_p < 1) or None
    - top_k: shortlist size (>= 1) or None
    - eos_id: token id to terminate on, or None to never early-stop
    - pad_id: tokenizer pad id used to build masks
    - max_new_tokens: number of tokens to generate
    """

    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_id: Optional[int] = None
    pad_id: int = 0
    max_new_tokens: int = 64


@dataclass
class VLMInputs:
    """Inputs for vision-language sampling.

    Expected shapes:
    - prompt_tokens: jnp.ndarray[int32] with shape [batch, seq]
    - vision: VisionEmbeddings or jnp.ndarray[float] with shape [vision_tokens, dim]
              or [batch, vision_tokens, dim]
    - grid_thw: jnp.ndarray[int32] with shape [batch, num_images, 3] (T, H, W)
    - image_pad_id: int placeholder token id used to inject vision tokens
    """

    prompt_tokens: jnp.ndarray
    vision: Union[VisionEmbeddings, jnp.ndarray]
    grid_thw: jnp.ndarray
    image_pad_id: int


@dataclass
class SampleResult:
    """Result of a sampling call.

    - tokens: jnp.ndarray[int32] with shape [batch, new_len]
    - logprobs: optional jnp.ndarray[float32] with shape [batch, new_len]
    - texts: decoded strings for each batch element (optional convenience)
    """

    tokens: jnp.ndarray
    logprobs: Optional[jnp.ndarray]
    texts: List[str] = field(default_factory=list)


@dataclass
class Batch:
    """Typed training batch for policy scoring.

    Expected shapes (with text targets defined as tokens[:, 1:]):
    - tokens: jnp.ndarray[int32] with shape [B, T]
    - token_mask: jnp.ndarray[int32] with shape [B, T-1] (mask over text targets)
    - cos/sin: RoPE caches matching text_input=tokens[:, :-1]
    - vision: VisionEmbeddings or jnp.ndarray with vision features
    - grid: jnp.ndarray[int32] with shape [B, num_images, 3]
    """

    tokens: jnp.ndarray
    token_mask: jnp.ndarray
    cos: jnp.ndarray
    sin: jnp.ndarray
    vision: Union[VisionEmbeddings, jnp.ndarray]
    grid: jnp.ndarray


@dataclass
class Rollout:
    """Typed rollout container for PPO/GRPO style updates.

    - tokens: padded sequences [N, T]
    - old_logprobs: sampling-time logprobs aligned with tokens (padded) [N, T]
    - returns: reward/advantage signals [N]
    - mask_targets: mask over text targets (tokens[:, 1:]) [N, T-1]
    - prompt_lens: prompt prefix lengths before actions [N]
    - action_lens: number of generated tokens per sample [N]
    - texts: decoded responses for previewing/logging
    """

    tokens: jnp.ndarray
    old_logprobs: jnp.ndarray
    returns: jnp.ndarray
    mask_targets: jnp.ndarray
    prompt_lens: jnp.ndarray
    action_lens: jnp.ndarray
    texts: List[str] = field(default_factory=list)


__all__ = [
    "SamplingConfig",
    "VLMInputs",
    "SampleResult",
    "Batch",
    "Rollout",
]

