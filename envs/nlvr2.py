from __future__ import annotations

from dataclasses import dataclass
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
class NLVR2Prompt:
    """Observation returned by `NLVR2Env.reset` for vision-language reasoning."""

    statement: str
    image_left: object  # PIL Image
    image_right: object  # PIL Image
    label: bool


@dataclass(frozen=True)
class NLVR2State(BaseState):
    dataset_idx: int
    statement: str
    image_left: object  # PIL Image
    image_right: object  # PIL Image
    label: bool

    def render(self) -> str:
        return f"Look at the two images. {self.statement} True or False?"


class NLVR2Env(BaseEnv):
    """NLVR2 visual reasoning environment with binary True/False labels.

    Each task presents two images and a natural language statement. The agent
    must determine whether the statement is True or False based on visual evidence.
    Rewards 1.0 for correct binary prediction, 0.0 otherwise.
    """

    def __init__(self, tokenizer, split: str = "validation"):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 128
        self.split = split
        self.dataset = self._load_dataset()
        self.num_tasks = len(self.dataset)
        self.eos_token_id = _resolve_eos_id(tokenizer)

    def _load_dataset(self) -> Sequence[dict]:
        from pathlib import Path
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        cache_dir = Path.home() / ".cache/huggingface/datasets"
        raw = load_dataset("HuggingFaceM4/NLVR2", split=self.split, cache_dir=cache_dir)
        parsed = []
        for item in raw:
            # NLVR2 returns PIL images; store them directly
            parsed.append(
                {
                    "statement": item["sentence"],
                    "image_left": item["left_image"],
                    "image_right": item["right_image"],
                    "label": bool(item["label"]),
                }
            )
        return parsed

    def reset(self, idx):
        item = self.dataset[idx % len(self.dataset)]
        state = NLVR2State(
            dataset_idx=idx % len(self.dataset),
            statement=item["statement"],
            image_left=item["image_left"],
            image_right=item["image_right"],
            label=item["label"],
        )
        obs = NLVR2Prompt(
            statement=item["statement"],
            image_left=item["image_left"],
            image_right=item["image_right"],
            label=item["label"],
        )
        return state, obs

    def step(self, state, action_tokens):
        if self.eos_token_id is not None:
            action_tokens = self.clean_action(action_tokens, self.eos_token_id)
        response_text = decode_tokens(self.tokenizer, action_tokens).lower().strip()

        predicted = self._parse_binary(response_text)
        reward = 1.0 if predicted == state.label else 0.0
        return state, [], reward, True, {"response": response_text, "predicted": predicted, "label": state.label}

    def _parse_binary(self, text: str) -> bool | None:
        """extremely simple binary parser"""
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
