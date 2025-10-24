"""Mock TUI for Attention, KV Cache, and mRoPE.

This module renders a non-interactive ASCII preview of a richer TUI that would
visualize internals of:
- Attention (per-layer/head grids)
- KV cache (per-layer occupancy and memory footprint)
- Multimodal RoPE (axes, sections, positions, scaling)
- Rollout summaries (prompt/action, rewards)

The goal is to demonstrate the layout and contract without requiring live JAX
hooks. It uses only stdlib and existing small helpers in `vlmrl.utils.tui`.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    # Optional: use existing ASCII helpers if present
    from vlmrl.utils.tui import TUI as _ASCII
except Exception:
    _ASCII = None  # fallback to local ramps


_RAMP = " .:-=+*#%@"


def _normalize(vals: Sequence[float]) -> List[float]:
    if not vals:
        return []
    mx = max(0.0, max(float(v) for v in vals))
    if mx <= 1e-12:
        return [0.0 for _ in vals]
    return [max(0.0, float(v) / mx) for v in vals]


def _render_grid_ascii(grid: Sequence[Sequence[float]]) -> Tuple[List[str], float]:
    ramp = _RAMP
    if _ASCII is not None:
        # Reuse existing helper if available
        try:
            return _ASCII._render_grid_ascii(grid)  # type: ignore[attr-defined]
        except Exception:
            pass
    if not grid:
        return [], 0.0
    flat = [float(x) for row in grid for x in row]
    mx = max(flat) if flat else 0.0
    if mx <= 1e-12:
        lines = ["".join(ramp[0] for _ in row) for row in grid]
        return lines, 0.0
    lines: List[str] = []
    for row in grid:
        s = []
        for v in row:
            idx = int(round((len(ramp) - 1) * max(0.0, float(v) / mx)))
            s.append(ramp[idx])
        lines.append("".join(s))
    return lines, float(mx)


def _bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        total = 1
    frac = max(0.0, min(1.0, float(current) / float(total)))
    filled = int(round(frac * width))
    return f"[{('#' * filled).ljust(width, '-')}] {current}/{total}"


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    if i == 0:
        return f"{int(x)}{units[i]}"
    return f"{x:.1f}{units[i]}"


@dataclass
class _TextSpec:
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_len: int
    dtype_bytes: int
    rope_section: Tuple[int, ...]
    rope_theta: float
    rope_scaling_type: Optional[str]
    rope_scaling_factor: Optional[float]


def _print_title(title: str) -> None:
    print(f"\n== {title} ==")


def _print_overview(step: int) -> None:
    reward_mean, reward_std = 2.451, 0.823
    approx_kl, clip_frac, loss = 0.0234, 0.142, 0.0512
    entropy, lr = 1.823, 1e-6
    speed, mem = "245 steps/min", "8.2/16.0 GB"
    print(
        f"Step {step} | reward {reward_mean:.3f}±{reward_std:.3f} | kl {approx_kl:.4f} | clip {clip_frac:.3f} | loss {loss:.4f}"
    )
    print(f"entropy {entropy:.3f} | lr {lr:g} | mem {mem} | speed {speed}")


def _print_rollout_prompt() -> None:
    prompt = "<image> Describe the scene."
    gen = "A fighter jet is taking off from a runway."
    tokens = ["<BOS>", "<image>", "Describe", "the", "scene", "."]
    credit = [0.01, 0.3, 0.45, 0.12, 0.7]  # aligns to tokens[1:]
    if _ASCII is not None:
        header = {
            "step": 245,
            "reward_mean": 2.451,
            "reward_std": 0.823,
            "approx_kl": 0.0234,
            "loss": 0.0512,
            "entropy": 1.823,
            "text_vision_split": (64, 256),
        }
        # small dummy head deltas for demo
        text_grid = [[random.random() for _ in range(16)] for _ in range(6)]
        vision_grid = [[random.random() for _ in range(8)] for _ in range(4)]
        _ASCII().render_step(header, tokens, credit, text_grid, vision_grid, flows=[(3, 5, 0.8), (5, 7, 0.6)])
    else:
        print("Prompt:", prompt)
        print("Generated:", gen)
        print("Tokens:", " ".join(tokens))


def _print_kv_cache(spec: _TextSpec) -> None:
    _print_title("KV Cache")
    cap = spec.max_len
    # Fake per-layer lengths ramping and small jitter per-batch
    layer_lengths = [min(cap, int(cap * (0.25 + 0.75 * i / max(1, spec.num_layers - 1)))) for i in range(spec.num_layers)]
    total_elems = 0
    for lid, L in enumerate(layer_lengths):
        elems = spec.num_heads * (L * spec.head_dim)
        total_elems += elems
        bar = _bar(L, cap)
        print(f"Layer {lid:02d} | heads={spec.num_heads:2d} | {bar}")
    per_tensor = total_elems * spec.dtype_bytes
    total_bytes = per_tensor * 2  # keys + values
    print(f"Totals: {spec.num_layers} layers × {spec.num_heads} heads × D{spec.head_dim} @ T{cap}")
    print(f"Memory (approx): {_fmt_bytes(total_bytes)} for K/V")


def _print_attention() -> None:
    _print_title("Attention Heads (sample)")
    grid = [[random.random() for _ in range(32)] for _ in range(10)]
    lines, mx = _render_grid_ascii(grid)
    if lines:
        print(f"Δ magnitude (max={mx:.2f})")
        for ln in lines:
            print(" " + ln)


def _print_mrope(spec: _TextSpec) -> None:
    _print_title("mRoPE")
    sec = ",".join(str(s) for s in spec.rope_section)
    print(
        f"axes={len(spec.rope_section)} sections=({sec}) | theta={spec.rope_theta:g} | scale={spec.rope_scaling_type or '-'}:{spec.rope_scaling_factor or '-'}"
    )
    # Show a tiny 2D HxW slice mapping to positions
    H, W = 6, 10
    pos_grid = [[(r * W + c) for c in range(W)] for r in range(H)]
    # Normalize for heatmap effect (just demo)
    mx = float(H * W - 1)
    heat = [[cell / mx for cell in row] for row in pos_grid]
    lines, _ = _render_grid_ascii(heat)
    print("pos(h,w) projected intensity:")
    for ln in lines:
        print(" " + ln)


def run_demo() -> None:
    random.seed(7)
    step = 245
    _print_title("Overview")
    _print_overview(step)
    _print_title("Rollout/Prompt")
    _print_rollout_prompt()
    # A plausible Qwen3‑VL text spec snapshot for illustration
    spec = _TextSpec(
        num_layers=28,
        num_heads=28,
        num_kv_heads=8,
        head_dim=128,
        max_len=256,
        dtype_bytes=2,  # bf16
        rope_section=(8, 24, 24),
        rope_theta=1000000.0,
        rope_scaling_type="linear",
        rope_scaling_factor=1.0,
    )
    _print_kv_cache(spec)
    _print_attention()
    _print_mrope(spec)


if __name__ == "__main__":
    run_demo()

