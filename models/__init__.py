"""Model backbones and conversion helpers exposed as a single namespace."""

from importlib import import_module

_SUBMODULES = (
    "qwen25vl",
    "qwen3vl",
)

__all__ = [*_SUBMODULES]


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(__all__)
