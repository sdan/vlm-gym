# TUI Usage Guide

## Quick Start

### Option 1: Standalone Demo

```bash
# Run the TUI demo with simulated data
python utils/tui_live.py
```

This runs a simulation showing what the TUI looks like with fake attention patterns and decode steps.

### Option 2: With Rollout (Real Model)

```bash
# Run rollout with TUI visualization
uv run python -m core.rollout \
  --model_dir /path/to/model \
  --env_name vision \
  --episodes 10 \
  --tui
```

**The `--tui` flag enables live visualization!**

## Full Example

```bash
# Example with all parameters
uv run python -m core.rollout \
  --model_dir ./checkpoints/qwen3vl-2b \
  --env_name vision \
  --episodes 20 \
  --batch_size 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_new_tokens 32 \
  --decode_impl step \
  --tui
```

## What You'll See

When you run with `--tui`, the terminal will clear and show:

```
╔════════════════════════════════════════╗
║ STEP 5  │ R: 0.85 │ KL: 0.02 │ ⚡ 142 tok/s ║
╚════════════════════════════════════════╝

DECODE LIVE                    Cache: 45/2048
Tok[12]: "Paris" p=0.89 ←━━━ SAMPLING NOW ⚡

  mRoPE → Q/K/V → Attention → KV Cache

ATTENTION FLOW (Last 6 Layers × 16 Heads)
L24  ░░▒█▓░░▒▓██░░▒░  │  ← Text self-attn
L25  ▒▓██░░▒░░▓█▒░░▒  │
L26  ░▒▓█████▓▒░░░░▒  │  ← Vision→Text ⚡
...

KV CACHE MEMORY
L24: [████░░░░░░░] 45/2048  180KB  ⚡ WRITE
...
```

The display updates in real-time as the model generates tokens!

## Controls

- **Ctrl+C** - Stop rollout and exit (cleanly shuts down TUI)
- The TUI auto-updates, no interaction needed

## Without TUI (Classic Mode)

If you omit `--tui`, you'll see the original text output:

```bash
uv run python -m core.rollout \
  --model_dir ./checkpoints/qwen3vl-2b \
  --env_name vision \
  --episodes 10
```

Output:
```
[episode 0] reward=0.850 done=True prompt='What is this?' response='A cat sitting on a mat' info_keys=none
[episode 1] reward=0.920 done=True prompt='Where is this?' response='Paris, France' info_keys=none
...
```

## Troubleshooting

### "Could not initialize TUI"

**Cause:** Missing dependencies or import errors

**Fix:**
```bash
# Make sure you're in the project directory
cd /Users/sdan/Developer/vlm-gym

# Try running the demo first
python utils/tui_live.py
```

### TUI not showing colors

**Cause:** Terminal doesn't support ANSI 256 colors

**Fix:**
```bash
# Set terminal type
export TERM=xterm-256color

# Then run again
uv run python -m core.rollout --model_dir ... --tui
```

### TUI is flickering

**Cause:** Terminal buffer issues

**Fix:**
- Use a better terminal (iTerm2, Alacritty, etc.)
- Or increase render delay in `utils/tui_live.py`:
  ```python
  time.sleep(0.1)  # Line 367, change from 0.05 to 0.1
  ```

### Model loading fails

**Cause:** Missing model checkpoint

**Fix:**
```bash
# Download a model first
# Or point to your existing checkpoint directory
ls checkpoints/  # Should show model files
```

## Advanced: Custom TUI

You can also use the TUI programmatically:

```python
from vlmrl.utils.tui_live import LiveTUI, DecodeState

# Create and start TUI
tui = LiveTUI(num_layers=28, num_heads=16)
tui.start()

# Your training/sampling loop
for step in range(100):
    # ... your code ...

    # Update TUI
    decode_state = DecodeState(
        step=step,
        token_text="hello",
        token_prob=0.95,
        cache_len=step,
        # ... other fields
    )
    tui.update_decode(decode_state)

# Cleanup
tui.stop()
```

## Performance Notes

- **Overhead:** ~1-2% (background thread)
- **Update rate:** 20 FPS max (can go lower on slow terminals)
- **Memory:** Minimal (~1MB for queue buffer)
- **GPU impact:** None (all visualization is CPU-side)

## Next Steps

1. **Try the demo**: `python utils/tui_live.py`
2. **Run with model**: `uv run python -m core.rollout --tui ...`
3. **Customize**: Edit `utils/tui_live.py` for your needs
4. **Integrate into training**: See `utils/TUI_README.md`

---

**Pro tip:** The TUI is designed to be hypnotic - watch the attention patterns pulse as the model thinks! 🧠⚡
