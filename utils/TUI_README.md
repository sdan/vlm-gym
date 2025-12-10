# Live TUI for Attention & KV Cache Visualization

## Overview

A **barebones, hypnotic TUI** that visualizes the raw computation flow through the Qwen3-VL model during training/inference. Shows the pulse of:

- **Attention mechanisms** (multi-head, grouped-query)
- **KV cache** state and updates
- **mRoPE** (multimodal rotary position embeddings)
- **Rollout/sampling** process
- **Vision-text interaction**

## Quick Start

```bash
# Run demo
python utils/tui_live.py

# Integrate into training
# See "Integration Guide" below
```

## What It Shows

### Single-Screen Layout

```
┏━ HEADER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Step | Reward | KL | Tokens/sec                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

DECODE LIVE                          Cache: 128/2048
  Current token being sampled

┌─ Computation Flow ─┐
│ mRoPE → Q/K/V → Attention → KV Cache Write    │
└────────────────────┘

ATTENTION FLOW (Last 6 Layers × 16 Heads)
  Real-time heatmap showing attention patterns

VISION INJECTION @ Token[3]
  Where vision embeddings merge with text

TOKEN CREDIT
  Per-token gradient × logprob importance

KV CACHE MEMORY
  Layer-wise cache utilization bars
```

### Visual Encoding

**Intensity Ramp:**
- `░` = 0.00-0.25 (low)
- `▒` = 0.25-0.50 (medium)
- `▓` = 0.50-0.75 (high)
- `█` = 0.75-1.00 (very high)

**Flashing Indicators** (at 5Hz):
- `⚡ ACTIVE` - Yellow: Active computation
- `⚡ WRITE` - Green: Cache write
- `⚡ SAMPLING` - Cyan: Token sampling

## Files

```
utils/
  tui_live.py              # Main TUI implementation
  tui_live_mockup.txt      # Visual mockup with 3 frames
  TUI_README.md            # This file
```

## Integration Guide

### 1. Import and Initialize

```python
from vlmrl.utils.tui_live import LiveTUI, DecodeState, TrainingState

# Create TUI instance
tui = LiveTUI(
    num_layers=28,
    num_heads=16,
    display_layers=6,
    flash_hz=5.0,
    max_cache=2048,
)

# Start background rendering thread
tui.start()
```

### 2. Hook into Sampling Loop

In `core/sampling.py`, after each decode step:

```python
# In _decode_loop_stepwise() or _decode_loop()
# After line 336 (token sampling)

decode_state = DecodeState(
    step=tok_step,
    token_id=int(tok),
    token_text=tokenizer.decode([int(tok)]),
    token_prob=float(logp),
    cache_len=int(cache_state.lengths[0]),
    cache_max=2048,
    active_layer=layer_id,  # Current layer
    active_head=0,  # You can track this
    mrope_t=int(grid_thw[0, 0, 0]),
    mrope_h=int(grid_thw[0, 0, 1]),
    mrope_w=int(grid_thw[0, 0, 2]),
    attention_pattern=None,  # Optional: extract from attn module
)
tui.update_decode(decode_state)
```

### 3. Hook into Training Loop

In `core/train.py` or `core/ppo.py`:

```python
# After PPO update step
training_state = TrainingState(
    step=step,
    reward_mean=float(rewards.mean()),
    reward_std=float(rewards.std()),
    kl=float(kl_divergence),
    loss=float(loss_value),
    tokens_per_sec=tokens_processed / elapsed_time,
    gpu_util=get_gpu_utilization(),  # Optional
)
tui.update_training(training_state)
```

### 4. Hook into Credit Assignment

After computing token credits (e.g., in rollout processing):

```python
# After episode complete
tokens = [tokenizer.decode([t]) for t in token_ids]
credits = compute_credits(advantages, logprobs)  # Your logic

tui.update_credit(tokens, credits)
```

### 5. Cleanup

```python
# At end of training
tui.stop()
```

## Advanced: Extract Attention Patterns

To show real attention heatmaps instead of random patterns, you need to capture intermediate values from the model.

### Option 1: JAX Hooks (Minimal Overhead)

```python
# In models/qwen3vl/model.py, MultiHeadAttention.__call__()

# After line 567 (before causal mask)
if hasattr(self, '_capture_attn') and self._capture_attn:
    # Store attention scores for visualization
    jax.debug.callback(
        lambda scores: tui.queue.put(('attn', scores)),
        attn_scores
    )
```

### Option 2: Return Auxiliary Data

Modify the model to optionally return attention scores:

```python
# In forward pass
def __call__(self, ..., return_attn=False):
    # ... existing code ...

    if return_attn:
        return out, cache, attn_scores
    return out, cache
```

Then extract in sampling loop:

```python
# If return_attn enabled
logits, cache, attn_scores = model.apply(...)
decode_state.attention_pattern = attn_scores[-6:, :]  # Last 6 layers
```

### Option 3: JAX vmap Tracing

Use `jax.experimental.host_callback` to extract intermediate activations without modifying model code.

## Performance Considerations

1. **Threading**: TUI runs in separate thread, non-blocking
2. **Queue size**: Limited to 100 updates to avoid memory buildup
3. **Render rate**: Capped at 20 FPS (50ms sleep)
4. **Flash rate**: 5 Hz by default (adjustable)
5. **Dropped frames**: If queue full, updates silently dropped

## Customization

### Change Colors

Edit color codes in `LiveTUI`:

```python
# ANSI 256-color palette
YELLOW = "\033[38;5;226m"  # Change to your preference
GREEN = "\033[38;5;46m"
CYAN = "\033[38;5;51m"
```

### Display Fewer/More Layers

```python
tui = LiveTUI(display_layers=8)  # Show last 8 layers
```

### Adjust Flash Speed

```python
tui = LiveTUI(flash_hz=10.0)  # Flash at 10 Hz
```

### Add More Metrics

Extend `TrainingState` and `DecodeState` dataclasses:

```python
@dataclass
class DecodeState:
    # ... existing fields ...
    attention_entropy: float = 0.0  # New field
```

Then update rendering in `_render_decode_step()`.

## Troubleshooting

**TUI not updating:**
- Check that `tui.start()` was called
- Verify queue is not full (increase `maxsize`)
- Ensure updates are being sent from training loop

**Flickering:**
- Increase sleep time in `_render_loop()`
- Use terminal with better buffer support

**Wrong attention patterns:**
- Verify you're extracting from correct layer indices
- Check array shapes match expected `[num_layers, num_heads]`

**Colors not showing:**
- Ensure terminal supports ANSI 256 colors
- Try `TERM=xterm-256color`

## Example Output

See `utils/tui_live_mockup.txt` for full mockup showing 3 frames:
1. Token being sampled
2. Attention firing
3. Cache write completing

## Technical Details

### Attention Visualization

- Shows **last N layers** (configurable)
- Each character = one attention head
- Intensity based on mean attention score for that head
- Flashes on high activity (>0.75)

### KV Cache Tracking

- One bar per layer
- Shows utilization as `current/max`
- Memory estimate: `tokens × heads × head_dim × 4 bytes`
- Status: `CLEARED | ACTIVE | ⚡ WRITE`

### mRoPE Display

- Shows 3D position: `T` (temporal), `H` (height), `W` (width)
- For text-only: `T=0, H=W=0`
- For vision: Reflects grid dimensions from `grid_thw`

### Token Credit

- Sorted by impact (highest first)
- Bar length proportional to credit value
- Flash on highest-impact token

## Next Steps

1. **Run demo**: `python utils/tui_live.py`
2. **Review mockup**: `cat utils/tui_live_mockup.txt`
3. **Integrate**: Follow integration guide above
4. **Customize**: Adjust colors, metrics, display

## Philosophy

> **Barebones but shows complexity**
> Watch the model's neurons fire in real-time. No clutter, just the raw pulse of computation.

The TUI is designed to be:
- **Minimal**: Single screen, no scrolling
- **Live**: Updates during decode (10-50Hz)
- **Hypnotic**: Flashing attention patterns create visual flow
- **Complete**: Shows every major computation step

It's not documentation—it's watching the model *think*.
