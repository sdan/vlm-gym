"""
Adapted from https://github.com/kvfrans/lmpo/tree/main/utils
"""
from argparse import ArgumentParser
import importlib
import os
import sys
import shutil
from typing import Optional

from flax.core import unfreeze
from huggingface_hub import snapshot_download

# Add project root to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlmrl.utils.checkpoint import Checkpoint


def _ensure_trailing_slash(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def _resolve_hf_dir(
    hf_dir: Optional[str],
    hf_repo: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Return a local directory containing the HF weights.

    If `hf_dir` is provided, use it. Otherwise, download a snapshot of
    `hf_repo` to the local cache using `huggingface_hub.snapshot_download`.
    """
    if hf_dir:
        return _ensure_trailing_slash(os.path.expanduser(hf_dir))

    snapshot_path = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        cache_dir=None if cache_dir is None else os.path.expanduser(cache_dir),
        local_files_only=False,
    )
    return _ensure_trailing_slash(snapshot_path)


def main() -> None:
    parser = ArgumentParser(description="Convert HF Qwen VL weights to a JAX checkpoint.")
    parser.add_argument(
        "--model_type",
        default="qwen3vl",
        choices=["qwen25vl", "qwen3vl"],
        help="Model type: qwen25vl or qwen3vl.",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Output directory for the converted checkpoint (will create if missing).",
    )
    parser.add_argument(
        "--hf_dir",
        default=None,
        help="Optional local HF snapshot directory. If omitted, downloads from --hf_repo.",
    )
    parser.add_argument(
        "--hf_repo",
        default=None,
        help="HF repo ID to download when --hf_dir is not provided. If omitted, uses default based on --model_type.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch, tag, or commit SHA).",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional HF cache directory override.",
    )

    args = parser.parse_args()

    # Set default repo based on model type if not provided
    if args.hf_repo is None:
        args.hf_repo = {
            "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen3vl": "Qwen/Qwen3-VL-4B-Instruct",
        }[args.model_type]

    ckpt_dir = _ensure_trailing_slash(os.path.expanduser(args.model_dir))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resolve or download HF snapshot
    hf_dir = _resolve_hf_dir(args.hf_dir, args.hf_repo, args.revision, args.cache_dir)
    print(f"Using HF snapshot at: {hf_dir}")

    # Dynamic import based on model type
    model_module = importlib.import_module(f"models.{args.model_type}")
    create_model_from_hf = model_module.create_model_from_hf

    # Build model and convert params
    _, params = create_model_from_hf(hf_dir)
    params = unfreeze(params)

    # Save converted params
    ckpt = Checkpoint(os.path.join(ckpt_dir, "params.pkl"), parallel=False)
    ckpt.params = params
    ckpt.save()

    # Copy relevant config/tokenizer files
    base_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "generation_config.json",
        "chat_template.json",
    ]

    # Add video_preprocessor_config.json for Qwen3-VL
    if args.model_type == "qwen3vl":
        base_files.append("video_preprocessor_config.json")

    for fname in base_files:
        src = os.path.join(hf_dir, fname)
        dst = os.path.join(ckpt_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied: {fname}")
        else:
            print(f"Warning: {fname} not found in HF directory; skipped copy.")


if __name__ == "__main__":
    main()
