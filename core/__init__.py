"""Core training loops, sampling glue, and shared types for this stack."""

from importlib import import_module

from .types import Batch, Rollout, SampleResult, SamplingConfig, VLMInputs

_SUBMODULES = (
    "batching",
    "eval",
    "kl",
    "minibatch",
    "policy",
    "ppo",
    "sampling",
    "train",
    "types",
)

__all__ = [
    "Batch",
    "Rollout",
    "SampleResult",
    "SamplingConfig",
    "VLMInputs",
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
