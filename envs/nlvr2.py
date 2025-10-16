"""NLVR2 visual reasoning environment with simple true/false rewards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from vlmrl.envs.base import BaseEnv, BaseState
from vlmrl.utils.vlm import decode_tokens


def _resolve_eos_id(tokenizer) -> int | None:
    """Best-effort lookup for tokenizer EOS id without binding to HF specifics."""
    if hasattr(tokenizer, "eos_token_id"):
        eos_id = getattr(tokenizer, "eos_token_id")
        if eos_id is not None:
            return int(eos_id)
    if hasattr(tokenizer, "get_eos_token_id"):
        return tokenizer.get_eos_token_id()
    return None


@dataclass(frozen=True)
class NLVR2Sample:
    """Single NLVR2 example with paired images and gold label."""

    statement: str
    image_left: Any  # PIL.Image.Image but optional import keeps deps lazy
    image_right: Any
    label: bool


@dataclass(frozen=True)
class NLVR2Prompt:
    """Observation returned by :meth:`NLVR2Env.reset`."""

    statement: str
    image_left: Any
    image_right: Any
    label: bool


@dataclass(frozen=True)
class NLVR2State(BaseState):
    """Environment state tracked across a single NLVR2 episode."""

    dataset_idx: int
    sample: NLVR2Sample

    def render(self) -> str:
        return f"Look at the two images. {self.sample.statement} True or False?"


class NLVR2Env(BaseEnv):
    """Tiny wrapper over Hugging Face's NLVR2 split for binary reasoning."""

    def __init__(self, tokenizer, split: str = "validation"):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 128
        self.split = split
        self.dataset = self._load_dataset(split)
        self.num_tasks = len(self.dataset)
        self.eos_token_id = _resolve_eos_id(tokenizer)

    def _load_dataset(self, split: str) -> Sequence[NLVR2Sample]:
        """Download (or reuse cached) HF dataset and coerce into light samples."""
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("Install datasets: pip install datasets") from exc

        cache_dir = Path.home() / ".cache/huggingface/datasets"
        hf_split = load_dataset("HuggingFaceM4/NLVR2", split=split, cache_dir=cache_dir)
        samples: list[NLVR2Sample] = []
        for row in hf_split:
            samples.append(
                NLVR2Sample(
                    statement=row["sentence"],
                    image_left=row["left_image"],
                    image_right=row["right_image"],
                    label=bool(row["label"]),
                )
            )
        return samples

    def reset(self, idx: int) -> tuple[NLVR2State, NLVR2Prompt]:
        """Return evaluation state and prompt for dataset index ``idx``."""
        sample = self.dataset[idx % len(self.dataset)]
        state = NLVR2State(dataset_idx=idx % len(self.dataset), sample=sample)
        prompt = NLVR2Prompt(
            statement=sample.statement,
            image_left=sample.image_left,
            image_right=sample.image_right,
            label=sample.label,
        )
        return state, prompt

    def step(self, state: NLVR2State, action_tokens: Sequence[int]):
        """Score a binary response using simple keyword parsing heuristics."""
        if self.eos_token_id is not None:
            action_tokens = self.clean_action(action_tokens, self.eos_token_id)
        response_text = decode_tokens(self.tokenizer, action_tokens).strip().lower()

        predicted = self._parse_binary(response_text)
        reward = 1.0 if predicted == state.sample.label else 0.0
        return state, [], reward, True, {
            "response": response_text,
            "predicted": predicted,
            "label": state.sample.label,
        }

    def _parse_binary(self, text: str) -> bool:
        """Extremely simple binary parser with yes/no fallback."""
        text = text.lower().strip()
        # check explicit true/false
        if "true" in text and "false" not in text:
            return True
        if "false" in text and "true" not in text:
            return False
        # check yes/no as fallback
        if "yes" in text and "no" not in text:
            return True
        if "no" in text and "yes" not in text:
            return False
        # default to False if ambiguous
        return False
