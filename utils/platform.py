import os
import platform
from typing import Any


def is_macos() -> bool:
    """Return True if running on macOS (Darwin)."""
    try:
        return platform.system() == "Darwin"
    except Exception:
        return False


def apply_macos_env() -> None:
    """Apply safe-by-default JAX/TF env vars for macOS Metal.

    Must be called before importing jax to take effect.
    """
    if not is_macos():
        return
    # Prefer Metal backend explicitly when available.
    os.environ.setdefault("JAX_PLATFORMS", "metal")
    # Quieter TF/XLA logs.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Avoid large up-front allocations on small GPUs.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Reduce staging overhead for DeepStack on constrained devices.
    os.environ.setdefault("VLMRL_SKIP_DEEPSTACK", "1")


def apply_macos_train_overrides(config: Any) -> None:
    """Adjust default training config for macOS local runs.

    These are defaults only; explicit CLI flags will still override them.
    """
    if not is_macos():
        return
    try:
        # Logging / run layout
        config.wandb_mode = "offline"
        config.save_dir = "runs/ppo-mac-tiny"

        # Duration / scale
        config.total_steps = 50
        config.batch_size = 1

        # Sampling / sequence
        config.low_memory = 1
        config.max_new_tokens = 32
        config.max_sequence_length = 256
        config.top_k = 256
        # Keep top_p as default 0.9
        config.vlm_max_pixels = 16_384

        # PPO / optimizer
        config.ppo_minibatch = 1
        config.ppo_epochs = 1
        config.learning_rate = 1e-6
        config.grad_checkpoint = 1
        config.entropy_coef = 0.0
    except Exception:
        # Best-effort: ignore if config is immutable or missing fields.
        pass

