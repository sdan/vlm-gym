"""Training trace utilities for per-layer/head update summaries and token credit.

This module is intentionally lightweight and JAX-agnostic on the host side.
It computes update magnitudes from parameter deltas and writes JSONL records
that a separate viewer can render (tokens on the left, head heatmaps on the right).

Usage:
  - Construct a TraceWriter once (e.g., in train.py) with a directory.
  - Call compute_update_heat with (old_params, new_params, model_spec) to get
    per-layer/per-head magnitudes for text and vision stacks.
  - Optionally compute basic token credits from PPO inputs and log a single
    minibatch for qualitative inspection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp


# -------------------------------
# JSONL writer
# -------------------------------


@dataclass
class TraceWriter:
    out_dir: str
    filename: str = "trace.jsonl"

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self._path = os.path.join(self.out_dir, self.filename)
        # Touch file
        try:
            with open(self._path, "a", encoding="utf-8"):
                pass
        except Exception:
            # If path cannot be created, keep a disabled writer
            self._path = ""

    def log(self, record: Dict[str, Any]) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Never let tracing crash training
            pass


# -------------------------------
# Update heat computation
# -------------------------------


def _per_head_norm_from_outdim(kernel: jnp.ndarray, num_heads: int, head_dim: int) -> List[float]:
    """Kernel shape [in_dim, num_heads*head_dim] -> List[H] L2 norms."""
    k = jnp.asarray(kernel, dtype=jnp.float32)
    in_dim, out_dim = k.shape
    if out_dim != num_heads * head_dim:
        return [0.0 for _ in range(int(num_heads))]
    k = k.reshape(in_dim, num_heads, head_dim)
    # Frobenius norm per head over (in_dim, head_dim)
    return [float(jnp.linalg.norm(k[:, h, :])) for h in range(int(num_heads))]


def _per_head_norm_from_indim(kernel: jnp.ndarray, num_heads: int, head_dim: int) -> List[float]:
    """Kernel shape [num_heads*head_dim, out_dim] -> List[H] L2 norms."""
    k = jnp.asarray(kernel, dtype=jnp.float32)
    in_dim, out_dim = k.shape
    if in_dim != num_heads * head_dim:
        return [0.0 for _ in range(int(num_heads))]
    k = k.reshape(num_heads, head_dim, out_dim)
    # Frobenius norm per head over (head_dim, out_dim)
    return [float(jnp.linalg.norm(k[h, :, :])) for h in range(int(num_heads))]


def _maybe_get(tree: dict, *keys: str):
    cur = tree
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _text_layer_head_updates(
    delta_params: dict,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> List[List[float]]:
    """Return [layers][heads] aggregate update magnitude for text attention.

    Aggregates q_proj, k_proj, v_proj, and o_proj magnitudes per head.
    """
    per_layer: List[List[float]] = []
    for li in range(int(num_layers)):
        layer = _maybe_get(delta_params, f"layers_{li}") or {}
        attn = _maybe_get(layer, "attn") or {}
        q = _maybe_get(attn, "q_proj", "kernel")
        k = _maybe_get(attn, "k_proj", "kernel")
        v = _maybe_get(attn, "v_proj", "kernel")
        o = _maybe_get(attn, "o_proj", "kernel")

        qn = _per_head_norm_from_outdim(q, num_heads, head_dim) if q is not None else [0.0] * num_heads
        # k/v may be grouped kv heads; we project to H-sized list regardless
        kn = _per_head_norm_from_outdim(k, num_heads, head_dim) if k is not None else [0.0] * num_heads
        vn = _per_head_norm_from_outdim(v, num_heads, head_dim) if v is not None else [0.0] * num_heads
        on = _per_head_norm_from_indim(o, num_heads, head_dim) if o is not None else [0.0] * num_heads

        layer_vals = [qn[h] + kn[h] + vn[h] + on[h] for h in range(int(num_heads))]
        per_layer.append(layer_vals)
    return per_layer


def _vision_layer_head_updates(
    delta_params: dict,
    depth: int,
    num_heads: int,
    head_dim: int,
) -> List[List[float]]:
    """Return [layers][heads] update magnitudes for vision attention heads.

    Uses the fused qkv kernel split and the proj kernel.
    """
    per_layer: List[List[float]] = []
    for li in range(int(depth)):
        blk = _maybe_get(delta_params, "visual", f"blocks_{li}") or {}
        attn = _maybe_get(blk, "attn") or {}
        qkv = _maybe_get(attn, "qkv", "kernel")
        proj = _maybe_get(attn, "proj", "kernel")
        if qkv is not None:
            qkv = jnp.asarray(qkv, dtype=jnp.float32)
            in_dim, out_dim = qkv.shape
            if out_dim == 3 * num_heads * head_dim:
                q, k, v = jnp.split(qkv, 3, axis=1)
                qn = _per_head_norm_from_outdim(q, num_heads, head_dim)
                kn = _per_head_norm_from_outdim(k, num_heads, head_dim)
                vn = _per_head_norm_from_outdim(v, num_heads, head_dim)
            else:
                qn = kn = vn = [0.0] * num_heads
        else:
            qn = kn = vn = [0.0] * num_heads
        on = _per_head_norm_from_indim(proj, num_heads, head_dim) if proj is not None else [0.0] * num_heads
        per_layer.append([qn[h] + kn[h] + vn[h] + on[h] for h in range(int(num_heads))])
    return per_layer


def compute_update_heat(
    old_params: Any,
    new_params: Any,
    model_spec: Any,
) -> Dict[str, Any]:
    """Compute per-layer/head update magnitudes for text and vision backbones.

    Returns a dict with:
      - text: {layers: [[.. per head ..], ...]}
      - vision: {layers: [[.. per head ..], ...]}
      - totals: {text_total, vision_total, total}
    """
    # Convert to host, unfreeze dicts if needed
    try:
        from flax.core import unfreeze

        oldd = unfreeze(jax.device_get(old_params))
        newd = unfreeze(jax.device_get(new_params))
    except Exception:
        oldd = jax.device_get(old_params)
        newd = jax.device_get(new_params)

    def tree_sub(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            keys = set(a.keys()) | set(b.keys())
            return {k: tree_sub(a.get(k), b.get(k)) for k in keys}
        try:
            if a is None and b is None:
                return 0.0
            if a is None:
                return jnp.asarray(b)
            if b is None:
                return jnp.asarray(a)
            return jnp.asarray(b) - jnp.asarray(a)
        except Exception:
            return 0.0

    delta = tree_sub(oldd, newd)

    text_layers = int(getattr(model_spec.text, "num_hidden_layers", 0) or 0)
    text_heads = int(getattr(model_spec.text, "num_attention_heads", 0) or 0)
    text_hdim = int(getattr(model_spec.text, "head_dim", 0) or 0)
    text_heat = _text_layer_head_updates(delta, text_layers, text_heads, text_hdim) if text_layers > 0 else []

    if getattr(model_spec, "vision", None) is not None:
        v_depth = int(model_spec.vision.depth)
        v_heads = int(model_spec.vision.num_heads)
        v_hdim = int(model_spec.vision.hidden_size // max(1, v_heads))
        vision_heat = _vision_layer_head_updates(delta, v_depth, v_heads, v_hdim)
    else:
        vision_heat = []

    # Totals for quick split view
    def _sum_2d(arr2d: Sequence[Sequence[float]]) -> float:
        return float(sum(float(x) for row in arr2d for x in row))

    t_total = _sum_2d(text_heat)
    v_total = _sum_2d(vision_heat)
    return {
        "text": {"layers": text_heat},
        "vision": {"layers": vision_heat},
        "totals": {"text_total": t_total, "vision_total": v_total, "total": t_total + v_total},
    }


# -------------------------------
# Token credit summary (simple)
# -------------------------------


def summarize_token_credit(
    tokens: jnp.ndarray,
    mask_targets: jnp.ndarray,
    advantages: jnp.ndarray,
    tokenizer=None,
    take_first_n: int = 48,
) -> Dict[str, Any]:
    """Return simple token-credit view for a small slice of the minibatch.

    - tokens: [B, T]
    - mask_targets: [B, T-1] mask of positions included in PPO loss
    - advantages: [B]
    We report the first example for clarity (configurable),
    attach detokenized text if a tokenizer is provided.
    """
    try:
        toks = jax.device_get(tokens)
        mask = jax.device_get(mask_targets)
        adv = jax.device_get(advantages)
    except Exception:
        toks, mask, adv = tokens, mask_targets, advantages

    if toks.shape[0] == 0:
        return {"example": None}
    b0 = 0
    ids = toks[b0]
    # Targets align to tokens[:, 1:]
    m = mask[b0]
    # Credit: broadcast per-token from sequence advantage
    credit = (adv[b0] * m).tolist()
    ids_list = ids.tolist()
    text = None
    if tokenizer is not None:
        try:
            text = tokenizer.decode(ids_list, skip_special_tokens=False)
        except Exception:
            text = None
    return {
        "example": {
            "token_ids": ids_list[:take_first_n],
            "token_credit": credit[: max(0, take_first_n - 1)],  # credit length is T-1
            "text": text,
        }
    }


__all__ = [
    "TraceWriter",
    "compute_update_heat",
    "summarize_token_credit",
]

