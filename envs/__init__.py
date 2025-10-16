"""RL environments, factories, and supporting glue used during training."""

from importlib import import_module

from .base import BaseEnv, BaseState, create_env

_SUBMODULES = (
    "base",
    "geospot",
    "nlvr2",
    "schedule",
    "vision_caption",
)

__all__ = [
    "BaseEnv",
    "BaseState",
    "create_env",
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
