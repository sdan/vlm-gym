from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vlmrl.envs.base import BaseEnv, BaseState
from vlmrl.utils.vlm import decode_tokens


def _resolve_eos_id(tokenizer) -> int | None:
    if hasattr(tokenizer, "eos_token_id"):
        eos_id = getattr(tokenizer, "eos_token_id")
        if eos_id is not None:
            return eos_id
    if hasattr(tokenizer, "get_eos_token_id"):
        return tokenizer.get_eos_token_id()
    return None


@dataclass(frozen=True)
class VisionPrompt:
    """Observation returned by `VisionCaptionEnv.reset` for vision-language pipelines."""

    question: str
    image_path: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class VisionCaptionState(BaseState):
    dataset_idx: int
    question: str
    image_path: str
    keywords: tuple[str, ...]

    def render(self) -> str:
        return self.question


class VisionCaptionEnv(BaseEnv):
    """Simple vision-language environment that rewards keyword hits.

    This environment is intentionally lightweight: each task pairs a question
    with one of the bundled demo images and rewards the agent for mentioning the
    associated keywords. It is not meant to be a rigorous benchmark, but rather
    a sanity-check environment for Qwen3-VL integration and PPO plumbing.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 128
        self.dataset = self._build_dataset()
        self.num_tasks = len(self.dataset)
        self.eos_token_id = _resolve_eos_id(tokenizer)

    def _build_dataset(self) -> Sequence[dict]:
        asset_root = Path(__file__).resolve().parents[1] / "imgs"
        return [
            {
                "prompt": "Describe the news content shown in the image.",
                "image": str(asset_root / "foxnews.png"),
                "keywords": ("news", "fox", "futures"),
            },
            {
                "prompt": "Give a concise description of the diagram.",
                "image": str(asset_root / "foxnews.png"),
                "keywords": ("nasdaq", "futures", "trading"),
            },
            {
                "prompt": "Explain what the Nasdaq futures are trading at in the image.",
                "image": str(asset_root / "foxnews.png"),
                "keywords": ("nasdaq", "futures", "trading"),
            },
            {
                "prompt": "Describe what is happening in this image.",
                "image": str(asset_root / "f35_takeoff.png"),
                "keywords": ("aircraft", "jet", "takeoff", "military"),
            },
            {
                "prompt": "What type of aircraft is shown in this image?",
                "image": str(asset_root / "f35_takeoff.png"),
                "keywords": ("f-35", "fighter", "jet", "aircraft"),
            },
            {
                "prompt": "Describe the ceremony shown in the image.",
                "image": str(asset_root / "japanese_ceremony.png"),
                "keywords": ("ceremony", "traditional", "wedding", "japanese"),
            },
            {
                "prompt": "What cultural event is taking place in this image?",
                "image": str(asset_root / "japanese_ceremony.png"),
                "keywords": ("ceremony", "traditional", "cultural", "ritual"),
            },
            {
                "prompt": "Describe the landscape in this image.",
                "image": str(asset_root / "mountain_landscape.png"),
                "keywords": ("mountain", "landscape", "nature", "scenic"),
            },
            {
                "prompt": "What natural features are visible in this image?",
                "image": str(asset_root / "mountain_landscape.png"),
                "keywords": ("mountain", "peak", "terrain", "wilderness"),
            },
            {
                "prompt": "Describe what the animal is doing in this image.",
                "image": str(asset_root / "panda_climbing.png"),
                "keywords": ("panda", "climbing", "bear", "animal"),
            },
            {
                "prompt": "What type of animal is shown in the image?",
                "image": str(asset_root / "panda_climbing.png"),
                "keywords": ("panda", "bear", "endangered", "wildlife"),
            },
            {
                "prompt": "Describe the aerial view in this image.",
                "image": str(asset_root / "stadium_aerial.png"),
                "keywords": ("stadium", "aerial", "sports", "venue"),
            },
            {
                "prompt": "What structure is visible from above in this image?",
                "image": str(asset_root / "stadium_aerial.png"),
                "keywords": ("stadium", "arena", "building", "architecture"),
            },
            {
                "prompt": "Describe the workspace setup in this image.",
                "image": str(asset_root / "coffee_laptop.png"),
                "keywords": ("laptop", "coffee", "workspace", "desk"),
            },
            {
                "prompt": "What items are visible on the desk in this image?",
                "image": str(asset_root / "coffee_laptop.png"),
                "keywords": ("laptop", "computer", "coffee", "work"),
            },
        ]

    def reset(self, idx):
        item = self.dataset[idx % len(self.dataset)]
        question = item["prompt"]
        state = VisionCaptionState(
            dataset_idx=idx % len(self.dataset),
            question=question,
            image_path=item["image"],
            keywords=tuple(item["keywords"]),
        )
        return state, VisionPrompt(question=question, image_path=item["image"], keywords=tuple(item["keywords"]))

    def step(self, state, action_tokens):
        if self.eos_token_id is not None:
            action_tokens = self.clean_action(action_tokens, self.eos_token_id)
        response_text = decode_tokens(self.tokenizer, action_tokens)

        reward = 0.0
        hits = 0
        for kw in state.keywords:
            if kw.lower() in response_text.lower():
                reward += 1.0
                hits += 1
        reward = reward / max(len(state.keywords), 1)
        return state, [], reward, True, {
            "response": response_text,
            "hits": hits,
            "num_keywords": len(state.keywords),
            # Include what the env is "looking for" to aid debugging
            "keywords": list(state.keywords),
            "question": state.question,
        }
