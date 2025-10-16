"""Utility helpers for checkpoints, configs, RNGs, and general training glue."""

from importlib import import_module

from .rng import RngSeq

_SUBMODULES = (
    "checkpoint",
    "configs",
    "hf_to_jax",
    "rng",
    "sharding",
    "train_state",
    "vlm",
    "wandb",
)

__all__ = [
    "RngSeq",
    *_SUBMODULES,
]


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(__all__)
