#!/usr/bin/env python3
"""Colab TPU setup script for vlm-gym low-memory training.

Quick start in a TPU Colab notebook:
    !curl -LsSf https://astral.sh/uv/install.sh | sh
    !git clone https://github.com/sdan/vlm-gym.git /content/vlm-gym
    %cd /content/vlm-gym
    !git checkout low-mem-gpu
    !python colab.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Execute shell command and stream output."""
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, text=True)


def setup_repo():
    """Clone and checkout low-mem-gpu branch."""
    repo_dir = Path("/content/vlm-gym")
    if not repo_dir.exists():
        run("git clone https://github.com/sdan/vlm-gym.git /content/vlm-gym")
    os.chdir(repo_dir)
    run("git fetch origin")
    run("git checkout low-mem-gpu", check=False)
    run("git pull origin low-mem-gpu", check=False)


def install_dependencies():
    """Install Python dependencies via uv."""
    # Install uv if not present
    uv_path = Path.home() / ".cargo" / "bin" / "uv"
    if not uv_path.exists():
        run("curl -LsSf https://astral.sh/uv/install.sh | sh")

    # Add uv to PATH for current session
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    if cargo_bin not in os.environ["PATH"]:
        os.environ["PATH"] = f"{cargo_bin}:{os.environ['PATH']}"

    # Create venv and install deps
    run("uv venv .venv --python 3.10")
    run("uv pip install -e .")


def configure_tpu():
    """Configure JAX for TPU runtime."""
    try:
        import jax
        print(f"JAX devices: {jax.devices()}")
        print(f"Device count: {jax.device_count()}")
        print(f"Local device count: {jax.local_device_count()}")

        # Verify TPU
        if jax.devices()[0].platform != "tpu":
            print("WARNING: No TPU detected. Running on CPU/GPU.")
        else:
            print(f"✓ TPU runtime ready: {jax.local_device_count()} cores")
    except ImportError:
        print("JAX not installed yet. Will verify after training.")


def download_checkpoint():
    """Download or verify Qwen3-VL checkpoint."""
    ckpt_dir = Path("checkpoints/qwen3vl_4b")
    if ckpt_dir.exists():
        print(f"✓ Checkpoint exists: {ckpt_dir}")
        return

    print("Checkpoint not found. You'll need to run conversion:")
    print("  python scripts/hf_to_jax.py --model_name Qwen/Qwen2-VL-2B-Instruct --output_dir checkpoints/qwen3vl_4b")
    print("\nFor quick testing, using a smaller model is recommended.")


def run_training():
    """Execute single rollout + 100-step training on TPU."""
    # Training flags optimized for TPU low-memory mode
    flags = {
        "low_memory": 1,
        "model_dir": "checkpoints/qwen3vl_4b",
        "save_dir": "runs/colab-tpu-test",
        "wandb_mode": "offline",
        "wandb_name": "colab-tpu-test",

        # Environment
        "env_name": "geospot",
        "env_split": "test",

        # Training duration
        "total_steps": 100,
        "batch_size": 4,  # Small for quick test
        "log_interval": 10,

        # Sampling
        "temperature": 0.7,
        "max_new_tokens": 32,  # Shorter for speed
        "vlm_max_pixels": 120_000,  # Low-mem cap

        # PPO tuned for TPU
        "ppo_minibatch": 8,  # Divisible by typical TPU core counts (8)
        "ppo_epochs": 1,
        "clip_epsilon": 0.2,

        # Optimizer
        "optimizer": "adafactor",
        "learning_rate": 1e-6,
        "max_grad_norm": 1.0,
    }

    flag_str = " ".join(f"--{k}={v}" for k, v in flags.items())
    cmd = f"python -m vlmrl.core.train {flag_str}"

    print("\n" + "="*80)
    print("Starting TPU training (1 rollout warmup + 100 steps)")
    print("="*80 + "\n")

    run(cmd)


def verify_results():
    """Check training outputs."""
    save_dir = Path("runs/colab-tpu-test")
    if (save_dir / "train_state.pkl").exists():
        print(f"\n✓ Training complete. Checkpoint saved to {save_dir}")
    else:
        print("\n✗ No checkpoint found. Check logs above for errors.")


def main():
    print("vlm-gym TPU Setup for Colab")
    print("="*80)

    # Check if running in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
    except ImportError:
        print("⚠ Not running in Colab. This script is optimized for Colab TPU runtime.")

    # Setup sequence
    setup_repo()
    install_dependencies()
    configure_tpu()
    download_checkpoint()

    # Run training
    run_training()
    verify_results()

    print("\n" + "="*80)
    print("Setup complete. To continue training:")
    print("  python -m vlmrl.core.train --resume_path runs/colab-tpu-test/train_state.pkl --total_steps 1000")
    print("="*80)


if __name__ == "__main__":
    main()
