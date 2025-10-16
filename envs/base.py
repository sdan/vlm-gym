"""Shared interfaces and factory helpers for RL-style VLM environments."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class BaseState:
    """Minimal state contract for environment episodes.

    Concrete states should surface whatever metadata downstream logging wants
    (dataset indices, prompts, curriculum stage, etc.) while remaining
    serialization-friendly.
    """

    def render(self) -> str:
        """Return a human-readable summary that can be logged or previewed."""
        raise NotImplementedError


class BaseEnv:
    """Lightweight RL wrapper expected by the PPO/GRPO training loops.

    Subclasses override :meth:`reset` and :meth:`step`, optionally tweaking the
    class attributes below to steer batching heuristics.
    """

    # Maximum tokens to request from the policy for a single action.
    tokens_per_action: int = 32
    # Step count at which the policy must emit an answer, or -1 to disable.
    force_answer_at: int = -1
    # Number of unique tasks in the dataset; -1 indicates unbounded streams.
    num_tasks: int = -1

    def reset(self, idx: int) -> Tuple[BaseState, Any]:
        """Return the initial state and model-facing observation for task ``idx``."""
        raise NotImplementedError

    def step(
        self,
        state: BaseState,
        action_tokens: Sequence[int],
    ) -> Tuple[BaseState, Sequence[int], float, bool, Dict[str, Any]]:
        """Advance one environment step using the model's token proposal."""
        raise NotImplementedError

    @staticmethod
    def clean_action(action_tokens: Sequence[int], end_token: int) -> List[int]:
        """Trim generations at the first EOS token if present."""
        tokens = list(action_tokens)
        try:
            index = tokens.index(end_token)
        except ValueError:
            return tokens
        return tokens[: index + 1]

    def step_list(
        self,
        states: Sequence[BaseState],
        actions: Sequence[Sequence[int]],
    ) -> Tuple[List[BaseState], List[Sequence[int]], List[float], List[bool], Dict[str, List[Any]]]:
        """Vectorized convenience wrapper used by the trainer's environment pool."""
        if len(states) != len(actions):
            raise ValueError(f"states/actions length mismatch: {len(states)} vs {len(actions)}")

        next_states: List[BaseState] = []
        next_outputs: List[Sequence[int]] = []
        rewards: List[float] = []
        dones: List[bool] = []
        info: Dict[str, List[Any]] = {}

        for state, action in zip(states, actions, strict=False):
            new_state, output_tokens, reward, is_done, extras = self.step(state, action)
            next_states.append(new_state)
            next_outputs.append(output_tokens)
            rewards.append(float(reward))
            dones.append(bool(is_done))
            for key, value in extras.items():
                info.setdefault(key, []).append(value)

        return next_states, next_outputs, rewards, dones, info


_ENV_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Binary reasoning
    "nlvr2": ("vlmrl.envs.nlvr2", "NLVR2Env"),
    "truefalse": ("vlmrl.envs.nlvr2", "NLVR2Env"),
    # Vision-language captioning demos
    "vision": ("vlmrl.envs.vision_caption", "VisionCaptionEnv"),
    "vision_caption": ("vlmrl.envs.vision_caption", "VisionCaptionEnv"),
    "caption": ("vlmrl.envs.vision_caption", "VisionCaptionEnv"),
    # Geolocation curriculum
    "geospot": ("vlmrl.envs.geospot", "GeospotEnv"),
}


def create_env(env_name: str, tokenizer, **env_kwargs) -> BaseEnv:
    """Factory helper to build known environments by identifier.

    The lookup is intentionally forgiving and supports a few aliases so the CLI
    and training configs can stay concise.
    """
    key = env_name.strip().lower()
    if key not in _ENV_REGISTRY:
        raise ValueError(f"Unknown environment name: {env_name!r}")

    module_path, attr = _ENV_REGISTRY[key]
    module = import_module(module_path)
    env_cls = getattr(module, attr)
    return env_cls(tokenizer, **env_kwargs)
