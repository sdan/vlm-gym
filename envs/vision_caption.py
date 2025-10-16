"""Tiny keyword-based captioning environment backed by local demo images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vlmrl.envs.base import BaseEnv, BaseState
from vlmrl.utils.vlm import decode_tokens


def _resolve_eos_id(tokenizer) -> int | None:
    """Attempt to reuse tokenizer EOS id when available."""
    if hasattr(tokenizer, "eos_token_id"):
        eos_id = getattr(tokenizer, "eos_token_id")
        if eos_id is not None:
            return int(eos_id)
    if hasattr(tokenizer, "get_eos_token_id"):
        return tokenizer.get_eos_token_id()
    return None


@dataclass(frozen=True)
class VisionCaptionSample:
    """Single captioning prompt with expected keyword set."""

    question: str
    image_path: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class VisionPrompt:
    """Observation returned by :meth:`VisionCaptionEnv.reset`."""

    question: str
    image_path: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class VisionCaptionState(BaseState):
    """Episode state for the lightweight captioning loop."""

    dataset_idx: int
    sample: VisionCaptionSample

    def render(self) -> str:
        return self.sample.question


class VisionCaptionEnv(BaseEnv):
    """Simple captioning environment scoring keyword coverage."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 128
        self.dataset = self._build_dataset()
        self.num_tasks = len(self.dataset)
        self.eos_token_id = _resolve_eos_id(tokenizer)

    def _build_dataset(self) -> Sequence[VisionCaptionSample]:
        """Bundle a handful of fixed prompts around the local demo assets."""
        asset_root = Path(__file__).resolve().parents[1] / "imgs"
        entries = [
            ("Describe the news content shown in the image.", "foxnews.png", ("news", "fox", "futures")),
            ("Give a concise description of the diagram.", "foxnews.png", ("nasdaq", "futures", "trading")),
            ("Explain what the Nasdaq futures are trading at in the image.", "foxnews.png", ("nasdaq", "futures", "trading")),
            ("Describe what is happening in this image.", "f35_takeoff.png", ("aircraft", "jet", "takeoff", "military")),
            ("What type of aircraft is shown in this image?", "f35_takeoff.png", ("f-35", "fighter", "jet", "aircraft")),
            ("Describe the ceremony shown in the image.", "japanese_ceremony.png", ("ceremony", "traditional", "wedding", "japanese")),
            ("What cultural event is taking place in this image?", "japanese_ceremony.png", ("ceremony", "traditional", "cultural", "ritual")),
            ("Describe the landscape in this image.", "mountain_landscape.png", ("mountain", "landscape", "nature", "scenic")),
            ("What natural features are visible in this image?", "mountain_landscape.png", ("mountain", "peak", "terrain", "wilderness")),
            ("Describe what the animal is doing in this image.", "panda_climbing.png", ("panda", "climbing", "bear", "animal")),
            ("What type of animal is shown in the image?", "panda_climbing.png", ("panda", "bear", "endangered", "wildlife")),
            ("Describe the aerial view in this image.", "stadium_aerial.png", ("stadium", "aerial", "sports", "venue")),
            ("What structure is visible from above in this image?", "stadium_aerial.png", ("stadium", "arena", "building", "architecture")),
            ("Describe the workspace setup in this image.", "coffee_laptop.png", ("laptop", "coffee", "workspace", "desk")),
            ("What items are visible on the desk in this image?", "coffee_laptop.png", ("laptop", "computer", "coffee", "work")),
        ]

        samples: list[VisionCaptionSample] = []
        for question, image_name, keywords in entries:
            samples.append(
                VisionCaptionSample(
                    question=question,
                    image_path=str(asset_root / image_name),
                    keywords=tuple(keywords),
                )
            )
        return samples

    def reset(self, idx: int) -> tuple[VisionCaptionState, VisionPrompt]:
        """Return state + observation pair for dataset index ``idx``."""
        sample = self.dataset[idx % len(self.dataset)]
        state = VisionCaptionState(dataset_idx=idx % len(self.dataset), sample=sample)
        prompt = VisionPrompt(
            question=sample.question,
            image_path=sample.image_path,
            keywords=sample.keywords,
        )
        return state, prompt

    def step(self, state: VisionCaptionState, action_tokens: Sequence[int]):
        """Reward normalized keyword coverage in the generated caption."""
        if self.eos_token_id is not None:
            action_tokens = self.clean_action(action_tokens, self.eos_token_id)
        response_text = decode_tokens(self.tokenizer, action_tokens)

        hits = sum(1 for kw in state.sample.keywords if kw.lower() in response_text.lower())
        reward = hits / max(len(state.sample.keywords), 1)
        return state, [], reward, True, {
            "response": response_text,
            "hits": hits,
            "num_keywords": len(state.sample.keywords),
            "keywords": list(state.sample.keywords),
            "question": state.sample.question,
        }
