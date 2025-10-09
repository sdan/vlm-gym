"""
From https://github.com/kvfrans/lmpo/tree/main/utils
"""
from argparse import ArgumentParser
import os
import shutil
from typing import Optional

from flax.core import unfreeze
from huggingface_hub import snapshot_download

from models.qwen25vl import create_model_from_hf
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
    parser = ArgumentParser(description="Convert HF Qwen2.5-VL weights to a JAX checkpoint.")
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
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HF repo ID to download when --hf_dir is not provided.",
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

    ckpt_dir = _ensure_trailing_slash(os.path.expanduser(args.model_dir))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resolve or download HF snapshot
    hf_dir = _resolve_hf_dir(args.hf_dir, args.hf_repo, args.revision, args.cache_dir)
    print(f"Using HF snapshot at: {hf_dir}")

    # Build model and convert params
    _, params = create_model_from_hf(hf_dir)
    params = unfreeze(params)

    # Save converted params
    ckpt = Checkpoint(os.path.join(ckpt_dir, "params.pkl"), parallel=False)
    ckpt.params = params
    ckpt.save()

    # Copy relevant config/tokenizer files
    for fname in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        src = os.path.join(hf_dir, fname)
        dst = os.path.join(ckpt_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {fname} not found in HF directory; skipped copy.")


if __name__ == "__main__":
    main()
