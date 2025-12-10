"""Live TUI for visualizing attention, KV cache, and mRoPE during training/inference.

Minimal, fast, and non-flashy visualization suitable for live runs.
All mock/demo data and flashing effects have been removed.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import List, Optional, Tuple

import numpy as np
import os
import colorsys
import math
import shutil
import re


@dataclass
class DecodeState:
    """State snapshot from a single decode step."""
    step: int = 0
    token_id: int = 0
    token_text: str = ""
    token_prob: float = 0.0
    cache_len: int = 0
    cache_max: int = 2048
    active_layer: int = 0
    active_head: int = 0
    mrope_t: int = 0
    mrope_h: int = 0
    mrope_w: int = 0
    attention_pattern: Optional[np.ndarray] = None  # [num_layers, num_heads]
    tokens_so_far: List[str] = field(default_factory=list)
    probs_so_far: List[float] = field(default_factory=list)


@dataclass
class TrainingState:
    """State snapshot from a training step."""
    step: int = 0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    kl: float = 0.0
    loss: float = 0.0
    tokens_per_sec: float = 0.0
    gpu_util: float = 0.0


@dataclass
class EpisodeView:
    """Compact episode summary for simple decoding view."""
    idx: int = 0
    reward: float = 0.0
    done: bool = False
    prompt: str = ""
    output: str = ""


class LiveTUI:
    """Barebones live TUI with flashing attention visualization.

    Single-screen display showing:
    - Current decode step with token being sampled
    - mRoPE → Q/K/V → Attention → KV cache flow
    - Attention heatmap (last 6 layers × 16 heads)
    - Vision injection overlay
    - Token credit assignments
    - KV cache memory bars
    """

    # ANSI control codes
    CLEAR_SCREEN = "\033[2J\033[H"
    ENTER_ALT = "\033[?1049h"  # Use terminal alternate screen buffer
    EXIT_ALT = "\033[?1049l"
    ENTER_ALT_FALLBACK = "\033[?47h"  # Older DEC private mode fallback
    EXIT_ALT_FALLBACK = "\033[?47l"
    CLEAR_TO_END = "\033[0J"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"

    def __init__(
        self,
        num_layers: int = 28,
        num_heads: int = 16,
        display_layers: int = 6,
        flash_hz: float = 5.0,  # kept for backward-compat; unused
        max_cache: int = 2048,
        simple_decode: bool = True,
        view_mode: Optional[str] = None,  # 'simple' | 'advanced' | 'merged'
        use_color: bool = True,
        truecolor: Optional[bool] = None,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.display_layers = display_layers
        self.max_cache = max_cache

        self.decode_state = DecodeState()
        self.training_state = TrainingState()

        self.update_queue: Queue = Queue(maxsize=100)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Token credit history (last episode)
        self.token_credits: List[Tuple[str, float]] = []
        # Streaming recent tokens during decode (token text, probability)
        self.recent_tokens: List[Tuple[str, float]] = []
        # Simple/advanced/merged view modes
        self.simple_decode = bool(simple_decode)
        if view_mode in ("simple", "advanced", "merged"):
            self.view_mode = str(view_mode)
        else:
            # Back-compat: map boolean to a mode
            self.view_mode = "simple" if self.simple_decode else "advanced"
        self.episodes_history: List[EpisodeView] = []

        # Color configuration for heatmaps
        self.use_color = bool(use_color) and self._detect_color_support()
        # Resolve truecolor preference: explicit arg overrides auto-detect
        if truecolor is None:
            self.truecolor = self._detect_truecolor_support()
        else:
            self.truecolor = bool(truecolor)
        # Respect NO_COLOR env var
        if os.getenv("NO_COLOR") is not None:
            self.use_color = False

    def start(self) -> None:
        """Start the TUI in a background thread."""
        if self.running:
            return
        self.running = True
        # Timers for uptime and CPU usage estimation
        self.start_time = time.time()
        self._cpu_last_wall: Optional[float] = None
        self._cpu_last_proc: Optional[float] = None
        self.cpu_percent: float = 0.0
        self.last_update_ts: float = 0.0
        self.thread = threading.Thread(target=self._render_loop, daemon=True)
        self.thread.start()
        # Switch to alternate screen buffer and hide cursor for a sticky view
        try:
            sys.stdout.write(self.ENTER_ALT)
            sys.stdout.flush()
        except Exception:
            try:
                sys.stdout.write(self.ENTER_ALT_FALLBACK)
                sys.stdout.flush()
            except Exception:
                pass
        sys.stdout.write(self.HIDE_CURSOR + self.CLEAR_SCREEN)
        sys.stdout.flush()

    def stop(self) -> None:
        """Stop the TUI and restore terminal."""
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        # Restore cursor and leave alternate screen buffer
        sys.stdout.write(self.SHOW_CURSOR)
        try:
            sys.stdout.write(self.EXIT_ALT)
        except Exception:
            try:
                sys.stdout.write(self.EXIT_ALT_FALLBACK)
            except Exception:
                pass
        sys.stdout.flush()

    def update_decode(self, state: DecodeState) -> None:
        """Update decode state (called from sampling loop)."""
        try:
            self.update_queue.put_nowait(("decode", state))
        except:
            pass  # Drop if queue full

    def update_training(self, state: TrainingState) -> None:
        """Update training state (called from PPO loop)."""
        try:
            self.update_queue.put_nowait(("training", state))
        except:
            pass

    def update_credit(self, tokens: List[str], credits: List[float]) -> None:
        """Update token credit assignments."""
        try:
            self.update_queue.put_nowait(("credit", (tokens, credits)))
        except:
            pass

    def update_episode(self, view: EpisodeView) -> None:
        """Append an episode summary for simple decode view."""
        try:
            self.update_queue.put_nowait(("episode", view))
        except:
            pass

    def _render_loop(self) -> None:
        """Main render loop (runs in background thread)."""
        last_render_update_ts: float = -1.0
        last_render_wall: float = 0.0
        while self.running:
            # Process updates from queue
            while not self.update_queue.empty():
                try:
                    msg_type, data = self.update_queue.get_nowait()
                    if msg_type == "decode":
                        # Track streaming tokens for Token scores panel
                        try:
                            if isinstance(data.token_text, str) and data.token_text:
                                prob = float(getattr(data, "token_prob", 0.0) or 0.0)
                                if math.isnan(prob) or prob < 0.0:
                                    prob = 0.0
                                elif prob > 1.0:
                                    prob = 1.0
                                self.recent_tokens.append((data.token_text, prob))
                                # Keep last 20
                                if len(self.recent_tokens) > 20:
                                    self.recent_tokens = self.recent_tokens[-20:]
                        except Exception:
                            pass
                        self.decode_state = data
                        self.last_update_ts = time.time()
                    elif msg_type == "training":
                        self.training_state = data
                        self.last_update_ts = time.time()
                    elif msg_type == "credit":
                        tokens, credits = data
                        # If credits look like log-probs (mostly <= 0), convert to probabilities
                        try:
                            arr = np.asarray(credits, dtype=np.float32)
                            if arr.size > 0 and (np.any(arr < 0.0) and not np.any(arr > 1.0)):
                                arr = np.exp(arr)
                                arr = np.clip(arr, 0.0, 1.0)
                            credits = arr.tolist()
                        except Exception:
                            pass
                        self.token_credits = list(zip(tokens, credits))
                        self.last_update_ts = time.time()
                    elif msg_type == "episode":
                        ev: EpisodeView = data
                        self.episodes_history.append(ev)
                        # Keep last 8 episodes
                        if len(self.episodes_history) > 8:
                            self.episodes_history = self.episodes_history[-8:]
                        self.last_update_ts = time.time()
                except:
                    break

            # Update CPU usage estimate
            try:
                now_w = time.perf_counter()
                now_p = time.process_time()
                if self._cpu_last_wall is not None and self._cpu_last_proc is not None:
                    dw = max(1e-6, now_w - self._cpu_last_wall)
                    dp = max(0.0, now_p - self._cpu_last_proc)
                    ncpu = max(1, (os.cpu_count() or 1))
                    self.cpu_percent = max(0.0, min(999.9, (dp / dw) * 100.0 / ncpu))
                self._cpu_last_wall = now_w
                self._cpu_last_proc = now_p
            except Exception:
                pass

            # Render throttled: at most every 0.5s, only if new data arrived.
            now = time.time()
            if (now - last_render_wall) >= 0.5 and self.last_update_ts != last_render_update_ts:
                last_render_update_ts = self.last_update_ts
                last_render_wall = now
                self._render_frame()
            # Failsafe heartbeat every 2s to keep CPU%/uptime fresh
            elif (now - last_render_wall) >= 2.0:
                last_render_wall = now
                self._render_frame()

            # Sleep to avoid burning CPU
            time.sleep(0.05)

    def _render_frame(self) -> None:
        """Render a single frame."""
        cols, rows = self._term_size()
        lines: List[str] = []

        # Header
        lines.append(self._render_header(cols))

        if self.view_mode == "simple":
            # Simple decode outputs view
            lines.append("")
            lines.append(self._truncate_ansi("DECODING OUTPUTS (last 8)", cols))
            lines.append(self._sep_line(cols))
            lines.extend(self._render_simple_decode())
        else:
            # Decode section
            lines.append("")
            lines.append(self._truncate_ansi(f"KV cache: {self.decode_state.cache_len}/{self.decode_state.cache_max} tokens", cols))
            lines.append(self._sep_line(cols))
            lines.append(self._render_decode_step())

            # Flow diagram
            lines.append("")
            lines.extend(self._render_flow_diagram())

            # Attention heatmap
            lines.append("")
            lines.append(self._truncate_ansi(f"Attention heads (last {self.display_layers} layers × {self.num_heads} heads)", cols))
            lines.append(self._sep_line(cols))
            lines.extend(self._render_attention_heatmap())

            # Token credit
            lines.append("")
            lines.append(self._truncate_ansi("Token scores", cols))
            lines.append(self._sep_line(cols))
            lines.extend(self._render_token_credit())

            # KV cache
            lines.append("")
            lines.append(self._truncate_ansi("KV cache usage", cols))
            lines.append(self._sep_line(cols))
            lines.extend(self._render_kv_cache())

        # If merged, append advanced sections after simple outputs
        if self.view_mode == "merged":
            lines.append("")
            lines.append(self._truncate_ansi(f"KV cache: {self.decode_state.cache_len}/{self.decode_state.cache_max} tokens", cols))
            lines.append(self._sep_line(cols))
            lines.append(self._render_decode_step())

            lines.append("")
            lines.extend(self._render_flow_diagram())

            lines.append("")
            lines.append(self._truncate_ansi(f"Attention heads (last {self.display_layers} layers × {self.num_heads} heads)", cols))
            lines.append(self._sep_line(cols))
            lines.extend(self._render_attention_heatmap())

            lines.append("")
            lines.append(self._truncate_ansi("Token scores", cols))
            lines.append(self._sep_line(cols))
            lines.extend(self._render_token_credit())

            lines.append("")
            lines.append(self._truncate_ansi("KV cache usage", cols))
            lines.append(self._sep_line(cols))
            lines.extend(self._render_kv_cache())

        # Footer
        lines.append("")
        lines.append(self._render_footer())

        # Clamp width and height to terminal; leave one line headroom
        lines = [self._truncate_ansi(ln, cols) for ln in lines]
        max_lines = max(1, rows - 1)
        lines = lines[:max_lines]

        # Write to screen (double buffering)
        # Full refresh, ensure trailing newline to avoid wrap-induced scroll
        output = self.CLEAR_SCREEN + "\n".join(lines) + "\n" + self.CLEAR_TO_END
        sys.stdout.write(output)
        sys.stdout.flush()

    def _render_simple_decode(self) -> List[str]:
        """Render minimal per-episode outputs with color coding."""
        lines: List[str] = []
        if not self.episodes_history:
            lines.append("No episodes yet.")
            return lines
        for ev in self.episodes_history:
            reward_col = self._fmt_reward(ev.reward)
            done_col = self._wrap_ansi(str(ev.done), 36 if ev.done else 90)  # cyan when done
            head_plain = f"[episode {ev.idx}] reward={ev.reward:.3f} done={ev.done}"
            head_col = head_plain
            try:
                head_col = (
                    f"[episode {ev.idx}] "
                    f"reward={reward_col} "
                    f"done={done_col}"
                )
            except Exception:
                pass
            lines.append(head_col)
            if ev.prompt:
                label = self._wrap_ansi("Prompt:", 36)  # cyan label
                lines.append(f"{label} {ev.prompt}")
            out_label = self._wrap_ansi("Output:", 35)  # magenta label
            lines.append(f"{out_label} {ev.output}")
            lines.append("")
        return lines

    def _render_header(self, cols: int) -> str:
        """Render top header with metrics."""
        ts = self.training_state
        # Uptime and last-update seconds
        up_secs = max(0.0, (time.time() - getattr(self, "start_time", time.time())))
        last_delta = max(0.0, (time.time() - getattr(self, "last_update_ts", 0.0))) if getattr(self, "last_update_ts", 0.0) > 0 else up_secs
        up_h = int(up_secs // 3600)
        up_m = int((up_secs % 3600) // 60)
        up_s = int(up_secs % 60)
        up_str = f"{up_h:02d}:{up_m:02d}:{up_s:02d}"
        cpu_str = f"CPU {getattr(self, 'cpu_percent', 0.0):.0f}%"
        last_str = f"Δ{last_delta:.1f}s"
        inner = max(10, cols - 2)
        top = "╔" + "═" * inner + "╗"
        content = (
            f"STEP {ts.step:<4}  │  Reward: {ts.reward_mean:.2f}  │  KL: {ts.kl:.3f}  │  "
            f"{ts.tokens_per_sec:.0f} tok/s   {cpu_str}   Up {up_str}   {last_str}"
        )
        mid = "║" + content.ljust(inner)[:inner] + "║"
        bot = "╚" + "═" * inner + "╝"
        return "\n".join([top, mid, bot])

    def _render_decode_step(self) -> str:
        """Render current decode step info."""
        ds = self.decode_state
        # Color token text and probability for readability
        tok_txt = ds.token_text.replace("\n", "⏎") if isinstance(ds.token_text, str) else ""
        p_str = f"{ds.token_prob:.2f}"
        p_col = self._colorize(max(0.0, min(1.0, float(ds.token_prob))), p_str)
        tok_col = self._wrap_ansi(f'"{tok_txt}"', 33)  # yellow token
        return f"Tok[{ds.step}]: {tok_col}  p={p_col}"

    def _render_flow_diagram(self) -> List[str]:
        """Render computation flow boxes."""
        ds = self.decode_state
        token_txt = ds.token_text if isinstance(ds.token_text, str) else ""
        token_txt = token_txt.replace("\n", "⏎")
        if len(token_txt) > 16:
            token_txt = token_txt[:15] + "…"
        tok_col = self._wrap_ansi(f'"{token_txt}"', 33)
        p_val = ds.token_prob
        try:
            p_val = float(p_val)
        except Exception:
            p_val = 0.0
        if math.isnan(p_val) or p_val < 0.0:
            p_val = 0.0
        elif p_val > 1.0:
            p_val = 1.0
        p_col = self._colorize(p_val, f"{p_val:.2f}")
        cache_max = max(0, int(getattr(ds, "cache_max", 0)))
        cache_str = f"{ds.cache_len}/{cache_max or '?'}"
        cols, _ = self._term_size()

        sampling_lines = [
            f"Token: {tok_col}",
            f"Step: {ds.step}   p: {p_col}",
            f"Cache: {cache_str}",
        ]

        rope_lines = [
            f"T: {self._wrap_ansi(str(ds.mrope_t), 36)}   H: {self._wrap_ansi(str(ds.mrope_h), 36)}",
            f"W: {self._wrap_ansi(str(ds.mrope_w), 36)}",
        ]
        qkv_lines = [
            f"L{ds.active_layer}  H{ds.active_head}",
            "Q [64d]",
            "K [64d]",
            "V [64d]",
        ]
        kv_action = "append" if cache_max == 0 or ds.cache_len < cache_max else "hold"
        kv_delta = "+1 token" if kv_action == "append" else "steady"
        kv_lines = [
            f"Action: {kv_action}",
            f"Len: {ds.cache_len}",
            f"Max: {cache_max or '?'}",
            f"Δ: {kv_delta}",
        ]

        # Very narrow terminals: fall back to compact textual layout.
        if cols < 48:
            compact_lines = [
                f"Token {tok_col}  step {ds.step}  p {p_col}",
                f"Cache {cache_str}  {kv_action}/{kv_delta}",
                f"RoPE T:{self._wrap_ansi(str(ds.mrope_t), 36)} "
                f"H:{self._wrap_ansi(str(ds.mrope_h), 36)} "
                f"W:{self._wrap_ansi(str(ds.mrope_w), 36)}",
                f"Layer L{ds.active_layer}  Head H{ds.active_head}",
            ]
            return compact_lines

        indent = 2
        gap = 3 if cols >= 60 else 2
        connector_mid = " → "
        connector_blank = " " * len(connector_mid)

        def make_box(title: str, content: List[str], width: int) -> Tuple[List[str], int]:
            width = max(4, min(width, cols - indent - 4))
            max_title_len = max(3, width - 1)
            if len(title) > max_title_len:
                if max_title_len >= 4:
                    title_display = title[: max_title_len - 1] + "…"
                else:
                    title_display = title[:max_title_len]
            else:
                title_display = title
            dash_count = max(0, width - len(title_display) - 1)
            color_code = 37 if self.use_color else None
            border_wrap = (lambda s: self._wrap_ansi(s, color_code)) if color_code else (lambda s: s)
            top = border_wrap(f"┌─ {title_display} " + "─" * dash_count + "┐")
            body: List[str] = []
            for line in content:
                trimmed = self._truncate_ansi(line, width)
                padded = self._pad_ansi(trimmed, width)
                body.append(f"│ {padded} │")
            bottom = border_wrap("└" + "─" * (width + 2) + "┘")
            return [top, *body, bottom], width

        def pad_box(lines: List[str], inner_width: int, target: int) -> None:
            blank = f"│ {' ' * inner_width} │"
            while len(lines) < target:
                lines.insert(-1, blank)

        def content_width(values: List[str]) -> int:
            if not values:
                return 12
            return max(len(self._ansi_strip(v)) for v in values)

        box_specs: List[Tuple[str, List[str], int]] = [
            ("Sampling", sampling_lines, 34),
            ("Positional (RoPE)", rope_lines, 34),
            ("Q/K/V", qkv_lines, 26),
            ("KV cache", kv_lines, 28),
        ]

        connector_width = len(connector_mid)
        min_inner = 12
        min_required = indent + sum((min_inner + 4) for _ in box_specs) + connector_width * (len(box_specs) - 1)

        horizontal_layout = False
        horizontal_widths: List[int] = []
        usable_width = cols - indent
        if cols >= min_required:
            preferred: List[int] = []
            for title, content, max_target in box_specs:
                needed_content = content_width(content)
                title_needed = len(title) + 1
                base = max(min_inner, needed_content + 2, title_needed)
                preferred.append(min(max_target, base))
            estimate = sum(w + 4 for w in preferred) + connector_width * (len(preferred) - 1)
            if estimate > usable_width:
                preferred = []
                for title, content, _ in box_specs:
                    needed_content = content_width(content)
                    title_needed = len(title) + 1
                    preferred.append(max(min_inner, needed_content, title_needed))
                estimate = sum(w + 4 for w in preferred) + connector_width * (len(preferred) - 1)
            if estimate <= usable_width:
                horizontal_layout = True
                horizontal_widths = preferred

        if horizontal_layout:
            boxes: List[Tuple[List[str], int]] = []
            for (title, content, _), inner_w in zip(box_specs, horizontal_widths):
                box_lines, actual_inner = make_box(title, content, inner_w)
                boxes.append((box_lines, actual_inner))
            max_height = max(len(b[0]) for b in boxes) if boxes else 0
            for idx, (box_lines, inner_w) in enumerate(boxes):
                pad_box(box_lines, inner_w, max_height)
                boxes[idx] = (box_lines, inner_w)

            mid_row = max_height // 2 if max_height else 0
            lines: List[str] = []
            for row in range(max_height):
                pieces: List[str] = []
                for idx, (box_lines, _) in enumerate(boxes):
                    pieces.append(box_lines[row])
                    if idx < len(boxes) - 1:
                        pieces.append(connector_mid if row == mid_row else connector_blank)
                lines.append((" " * indent) + "".join(pieces))
            return lines

        def clamp_inner(max_target: int) -> int:
            inner_cap = max(12, cols - indent - 4)
            return max(12, min(max_target, inner_cap))

        sample_inner = clamp_inner(32)
        rope_inner = clamp_inner(32)
        pair_available = cols - indent - gap - 8

        sampling_box, sample_inner = make_box("Sampling", sampling_lines, width=sample_inner)
        rope_box, rope_inner = make_box("Positional (RoPE)", rope_lines, width=rope_inner)

        if pair_available >= 26:
            pair_inner = max(13, min(18, pair_available // 2))
            qkv_box, q_inner = make_box("Q/K/V", qkv_lines, width=pair_inner)
            kv_box, kv_inner = make_box("KV cache", kv_lines, width=pair_inner)
            two_column = True
        else:
            stack_inner = clamp_inner(24)
            qkv_box, q_inner = make_box("Q/K/V", qkv_lines, width=stack_inner)
            kv_box, kv_inner = make_box("KV cache", kv_lines, width=stack_inner)
            two_column = False

        lines: List[str] = []
        lines.extend("  " + ln for ln in sampling_box)
        lines.append("  ↓")
        lines.append("")
        lines.extend("  " + ln for ln in rope_box)
        lines.append("  ↓")
        lines.append("")
        if two_column:
            from itertools import zip_longest

            target_height = max(len(qkv_box), len(kv_box))
            pad_box(qkv_box, q_inner, target_height)
            pad_box(kv_box, kv_inner, target_height)
            spacer = " " * gap
            for q_line, kv_line in zip_longest(qkv_box, kv_box, fillvalue=""):
                lines.append("  " + q_line + spacer + kv_line)
        else:
            lines.extend("  " + ln for ln in qkv_box)
            lines.append("  ↓")
            lines.extend("  " + ln for ln in kv_box)

        return lines

    def _render_attention_heatmap(self) -> List[str]:
        """Render attention pattern heatmap."""
        lines = []

        # Use actual attention pattern if available; otherwise indicate no data
        attn = self.decode_state.attention_pattern
        if attn is None or attn.ndim != 2 or attn.shape[1] != self.num_heads:
            lines.append("No attention data.")
            return lines

        # Determine which layers to display (last N layers)
        start_layer = max(0, self.num_layers - self.display_layers)
        rows = min(self.display_layers, attn.shape[0])

        for i in range(rows):
            layer_idx = start_layer + i
            row_vals = attn[i]
            if self.use_color:
                blocks = "".join([
                    self._colorize(float(v), self._intensity_char(float(v))) for v in row_vals
                ])
            else:
                blocks = "".join([self._intensity_char(float(v)) for v in row_vals])
            cols, _ = self._term_size()
            lines.append(self._truncate_ansi(f"L{layer_idx:<2}  {blocks}", cols))

        return lines

    def _render_token_credit(self) -> List[str]:
        """Render token scores. Prefer streaming last tokens; fallback to episode credits."""
        lines: List[str] = []

        # Prefer streaming tokens (chronological, last 10)
        if self.recent_tokens:
            tail = self.recent_tokens[-10:]
            cols, _ = self._term_size()
            for i, (tok, prob) in enumerate(tail):
                bar_len = max(0, min(10, int(prob * 10)))
                bar_col = "".join(self._colorize(prob, "█") for _ in range(bar_len))
                bar = bar_col if self.use_color else ("█" * bar_len)
                prefix = "→ " if i == len(tail) - 1 else "  "
                p_str = self._colorize(prob, f"{prob:.2f}") if self.use_color else f"{prob:.2f}"
                tok_disp = tok.replace("\n", "⏎") if isinstance(tok, str) else ""
                tok_col = self._wrap_ansi(f"{tok_disp:<12}", 33)  # yellow token label
                lines.append(self._truncate_ansi(f"{prefix}{tok_col} {bar:<10} {p_str}", cols))
            return lines

        # Fallback: top tokens by credit (assumed probabilities)
        if not self.token_credits:
            lines.append("No token credit data.")
            return lines

        credits = sorted(self.token_credits, key=lambda x: x[1], reverse=True)[:5]
        cols, _ = self._term_size()
        for token, credit in credits:
            v = max(0.0, min(1.0, float(credit)))
            bar_len = max(0, min(10, int(v * 10)))
            bar_col = "".join(self._colorize(v, "█") for _ in range(bar_len))
            bar = bar_col if self.use_color else ("█" * bar_len)
            tok_disp = token.replace("\n", "⏎") if isinstance(token, str) else ""
            tok_col = self._wrap_ansi(f"{tok_disp:<12}", 33)
            v_str = self._colorize(v, f"{v:.2f}") if self.use_color else f"{v:.2f}"
            lines.append(self._truncate_ansi(f" {tok_col} {bar:<10} {v_str}", cols))

        return lines

    def _render_kv_cache(self) -> List[str]:
        """Render KV cache memory bars."""
        lines = []

        # Show last N layers
        start_layer = max(0, self.num_layers - self.display_layers)
        cache_len = self.decode_state.cache_len

        for i in range(self.display_layers):
            layer_idx = start_layer + i

            # Progress bar
            progress = int((cache_len / self.max_cache) * 30)
            if self.use_color and self.max_cache > 0:
                ratio = max(0.0, min(1.0, cache_len / max(1, self.max_cache)))
                bar_col = "".join(self._colorize(ratio, "█") for _ in range(progress))
                bar = bar_col + "░" * (30 - progress)
            else:
                bar = "█" * progress + "░" * (30 - progress)

            # Memory estimate (rough)
            mem_kb = int((cache_len * self.num_heads * 64 * 4) / 1024)  # 4 bytes per float

            # Status (no flashing)
            status = "ACTIVE" if cache_len > 0 else "CLEARED"

            cols, _ = self._term_size()
            lines.append(self._truncate_ansi(f"L{layer_idx:<2}: [{bar}] {cache_len}/{self.max_cache}  {mem_kb}KB  {status}", cols))

        return lines

    def _render_footer(self) -> str:
        """Render bottom controls."""
        gpu = self.training_state.gpu_util
        gpu_str = f"GPU: {gpu:.0f}%"
        gpu_col = self._wrap_ansi(gpu_str, 36)
        return "[P] Pause │ [S] Step │ [L] Layers │ [Q] Quit" + " " * 24 + gpu_col

    def _intensity_char(self, value: float) -> str:
        """Map 0-1 value to intensity character."""
        if value < 0.25:
            return "░"
        elif value < 0.50:
            return "▒"
        elif value < 0.75:
            return "▓"
        else:
            return "█"

    # No flashing/emoji decorations

    def _detect_color_support(self) -> bool:
        """Best-effort detection of color-capable TTY."""
        try:
            # Basic checks: TTY present and not a 'dumb' terminal
            is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
            term = os.getenv("TERM", "")
            if not is_tty or term.lower() == "dumb":
                return False
            return True
        except Exception:
            return False

    def _detect_truecolor_support(self) -> bool:
        """Detect truecolor capability via COLORTERM hints."""
        ct = (os.getenv("COLORTERM", "") or "").lower()
        return ("truecolor" in ct) or ("24bit" in ct)

    def _colorize(self, value: float, ch: str) -> str:
        """Wrap a character with a color based on [0,1] value.

        Uses HSV gradient from blue (low) to red (high). Falls back to
        8-color ANSI bands if truecolor is not available.
        """
        if not self.use_color:
            return ch
        v = 0.0 if (value is None or math.isnan(value)) else max(0.0, min(1.0, float(value)))
        if self.truecolor:
            # HSV: h ∈ [0.0, 0.66] with 0.66≈blue, 0.0=red
            h = (2.0 / 3.0) * (1.0 - v)
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            R = int(255 * r)
            G = int(255 * g)
            B = int(255 * b)
            return f"\033[38;2;{R};{G};{B}m{ch}\033[0m"
        else:
            # 8/16-color fallback bands: blue → cyan → green → yellow → red
            if v < 0.20:
                code = 34  # blue
            elif v < 0.40:
                code = 36  # cyan
            elif v < 0.60:
                code = 32  # green
            elif v < 0.80:
                code = 33  # yellow
            else:
                code = 31  # red
            return f"\033[{code}m{ch}\033[0m"

    def _wrap_ansi(self, s: str, code: int) -> str:
        """Wrap a string with an ANSI SGR color code if enabled."""
        if not self.use_color:
            return s
        try:
            return f"\033[{int(code)}m{s}\033[0m"
        except Exception:
            return s

    def _fmt_reward(self, r: float) -> str:
        """Color-code reward: red (neg), yellow (low), green (high)."""
        try:
            v = float(r)
        except Exception:
            return str(r)
        if not self.use_color:
            return f"{v:.3f}"
        if v < 0.0:
            code = 31  # red
        elif v < 0.5:
            code = 33  # yellow
        else:
            code = 32  # green
        return self._wrap_ansi(f"{v:.3f}", code)

    def _term_size(self) -> Tuple[int, int]:
        try:
            sz = shutil.get_terminal_size(fallback=(80, 24))
            return int(sz.columns), int(sz.lines)
        except Exception:
            return 80, 24

    def _sep_line(self, width: int) -> str:
        return "─" * max(1, width)

    def _two_col(self, left: str, right: str, width: int) -> str:
        left_s = self._truncate_ansi(left, width)
        right_s = self._truncate_ansi(right, width)
        lw = len(self._ansi_strip(left_s))
        rw = len(self._ansi_strip(right_s))
        if lw + 1 + rw <= width:
            return left_s + " " * (width - (lw + rw)) + right_s
        # Not enough space: truncate left side
        avail_left = max(0, width - 1 - rw)
        return self._truncate_ansi(self._ansi_strip(left)[:avail_left] + " " + self._ansi_strip(right), width)

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _ansi_strip(self, s: str) -> str:
        try:
            return self._ANSI_RE.sub("", s)
        except Exception:
            return s

    def _pad_ansi(self, s: str, width: int) -> str:
        """Right-pad a string that may contain ANSI codes to a visible width."""
        stripped = self._ansi_strip(s)
        if len(stripped) > width:
            s = self._truncate_ansi(s, width)
            stripped = self._ansi_strip(s)
        padding = max(0, width - len(stripped))
        if padding <= 0:
            return s
        return s + (" " * padding)

    def _truncate_ansi(self, s: str, width: int) -> str:
        """Truncate string to width counting printable chars; keep ANSI codes intact."""
        if width <= 0 or not s:
            return ""
        out = []
        count = 0
        i = 0
        had_ansi = False
        n = len(s)
        while i < n and count < width:
            ch = s[i]
            if ch == "\x1b":
                # parse until 'm'
                j = i + 1
                while j < n and s[j] != 'm':
                    j += 1
                j = min(j + 1, n)
                out.append(s[i:j])
                had_ansi = True
                i = j
                continue
            out.append(ch)
            count += 1
            i += 1
        # If truncated and colored, ensure reset
        res = "".join(out)
        if i < n and had_ansi and not res.endswith("\x1b[0m"):
            res += "\x1b[0m"
        return res


__all__ = ["LiveTUI", "DecodeState", "TrainingState", "EpisodeView"]
