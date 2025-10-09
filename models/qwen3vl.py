"""Qwen3-VL vision-language utilities.

This module wraps around Hugging Face's ``Qwen3VLMoeForConditionalGeneration``

TODO: Move to jax once <30b weights are supported.

We rely on ``qwen_vl_utils.process_vision_info`` (distributed alongside the
official Qwen repos) to normalise image/video inputs before handing them to the
``AutoProcessor``.
"""
from __future__ import annotations

import itertools
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration, Trainer


logger = logging.getLogger(__name__)  # keep logs quiet by default

_PATH_ROOT = Path(__file__).resolve().parents[1]
_QWEN_UTILS_SRC = (_PATH_ROOT / ".." / "qwen3vl" / "qwen-vl-utils" / "src").resolve()
if _QWEN_UTILS_SRC.exists() and str(_QWEN_UTILS_SRC) not in sys.path:
    sys.path.append(str(_QWEN_UTILS_SRC))

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except ImportError as exc:  # pragma: no cover - defensive import guard
    raise ImportError(
        "qwen_vl_utils is required for Qwen3-VL integration. Ensure the "
        "qwen-vl-utils package is installed or the ../qwen3vl/qwen-vl-utils "
        "directory is available."
    ) from exc


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"


@dataclass
class TextBackboneSpec:
    """Key hyperparameters for the text decoder stack."""

    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    rope_theta: float
    rope_scaling_type: Optional[str]
    rope_scaling_factor: Optional[float]
    rms_norm_eps: float
    attention_bias: Optional[bool]
    max_position_embeddings: Optional[int]
    num_experts: Optional[int]
    num_experts_per_tok: Optional[int]
    qk_norm: Optional[bool]
    qk_norm_epsilon: Optional[float]


@dataclass
class VisionBackboneSpec:
    """Configuration summary for the vision encoder."""

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    out_hidden_size: int
    num_position_embeddings: Optional[int]
    deepstack_visual_indexes: Sequence[int]


@dataclass
class Qwen3VLStructure:
    """High-level view of the combined VLM architecture."""

    vocab_size: int
    pad_token_id: Optional[int]
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    tie_word_embeddings: bool
    text: TextBackboneSpec
    vision: VisionBackboneSpec


@dataclass
class VLMModelArguments:
    """Subset of HF model arguments that matter for Qwen3-VL finetuning."""

    model_name_or_path: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    tune_mm_llm: bool = False
    tune_mm_mlp: bool = False
    tune_mm_vision: bool = False


@dataclass
class DatasetSpec:
    """Represents a single supervised dataset source."""

    annotation_path: str
    data_path: Optional[str] = None
    sampling_rate: float = 1.0


@dataclass
class VLMDatasetArguments:
    """Minimal dataset knobs carried over from the standalone trainer."""

    dataset_use: str = ""
    data_flatten: bool = False
    data_packing: bool = False
    base_interval: int = 2
    max_pixels: int = 28 * 28 * 576
    min_pixels: int = 28 * 28 * 16
    video_max_frames: Optional[int] = 8
    video_min_frames: Optional[int] = 4
    video_max_pixels: int = 1024 * 28 * 28
    video_min_pixels: int = 256 * 28 * 28
    video_fps: float = 2.0
    model_type: str = "qwen3vl"
    local_rank: int = 0

    def parsed_datasets(self) -> list[DatasetSpec]:
        """Return dataset specs parsed from the comma-separated manifest string."""

        specs: list[DatasetSpec] = []
        if not self.dataset_use:
            return specs
        for raw_entry in self.dataset_use.split(","):
            entry = raw_entry.strip()
            if not entry:
                continue
            if "@" in entry:
                path_part, rate_part = entry.rsplit("@", 1)
                try:
                    sampling_rate = float(rate_part)
                except ValueError as exc:  # pragma: no cover - defensive guard
                    raise ValueError(f"Invalid sampling rate in dataset spec '{entry}'") from exc
            else:
                path_part = entry
                sampling_rate = 1.0
            annotation_path = str(Path(path_part).expanduser().resolve())
            specs.append(
                DatasetSpec(
                    annotation_path=annotation_path,
                    data_path=str(Path(annotation_path).parent),
                    sampling_rate=sampling_rate,
                )
            )
        return specs


@dataclass
class VLMTrainingArguments(transformers.TrainingArguments):
    """TrainingArguments extension with VLM-specific LR groups."""

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded."},
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None


def _maybe_get(mapping: Any, name: str, default: Optional[Any] = None) -> Any:
    return getattr(mapping, name, default)  # tiny getattr helper


def describe_qwen3_vl(config_or_model: Any) -> Qwen3VLStructure:
    """Return a compact summary of the Qwen3-VL architecture.

    ``config_or_model`` can be either an instantiated model or a config object.
    The function mirrors the structural breakdown we keep for the text-only
    Qwen3 models, which makes it easier to compare the two implementations side
    by side.
    """

    config = getattr(config_or_model, "config", config_or_model)  # accept model or config
    text_cfg = _maybe_get(config, "text_config", config)
    vision_cfg = _maybe_get(config, "vision_config")
    if vision_cfg is None:
        raise ValueError("Qwen3-VL config is expected to expose `vision_config`.")

    rope_scaling = _maybe_get(text_cfg, "rope_scaling") or {}  # optional rope scaling
    if isinstance(rope_scaling, dict):
        rope_scaling_type = rope_scaling.get("type")
        rope_scaling_factor = rope_scaling.get("factor") or rope_scaling.get("finetuned_factor")
    else:
        rope_scaling_type = None
        rope_scaling_factor = None

    text_spec = TextBackboneSpec(
        hidden_size=_maybe_get(text_cfg, "hidden_size", 0),
        head_dim=_maybe_get(text_cfg, "head_dim", 0),
        num_attention_heads=_maybe_get(text_cfg, "num_attention_heads", 0),
        num_key_value_heads=_maybe_get(text_cfg, "num_key_value_heads", 0),
        num_hidden_layers=_maybe_get(text_cfg, "num_hidden_layers", 0),
        intermediate_size=_maybe_get(text_cfg, "intermediate_size", 0),
        rope_theta=float(_maybe_get(text_cfg, "rope_theta", 1.0)),
        rope_scaling_type=rope_scaling_type,
        rope_scaling_factor=rope_scaling_factor,
        rms_norm_eps=float(_maybe_get(text_cfg, "rms_norm_eps", 1e-6)),
        attention_bias=_maybe_get(text_cfg, "attention_bias"),
        max_position_embeddings=_maybe_get(text_cfg, "max_position_embeddings"),
        num_experts=_maybe_get(text_cfg, "num_experts"),
        num_experts_per_tok=_maybe_get(text_cfg, "num_experts_per_tok"),
        qk_norm=_maybe_get(text_cfg, "qk_norm"),
        qk_norm_epsilon=_maybe_get(text_cfg, "qk_norm_epsilon"),
    )

    deepstack_idx = tuple(_maybe_get(vision_cfg, "deepstack_visual_indexes", ()))
    vision_spec = VisionBackboneSpec(
        depth=_maybe_get(vision_cfg, "depth", 0),
        hidden_size=_maybe_get(vision_cfg, "hidden_size", 0),
        intermediate_size=_maybe_get(vision_cfg, "intermediate_size", 0),
        num_heads=_maybe_get(vision_cfg, "num_heads", 0),
        patch_size=_maybe_get(vision_cfg, "patch_size", 1),
        spatial_merge_size=_maybe_get(vision_cfg, "spatial_merge_size", 1),
        temporal_patch_size=_maybe_get(vision_cfg, "temporal_patch_size", 1),
        out_hidden_size=_maybe_get(vision_cfg, "out_hidden_size", 0),
        num_position_embeddings=_maybe_get(vision_cfg, "num_position_embeddings"),
        deepstack_visual_indexes=deepstack_idx,
    )

    return Qwen3VLStructure(
        vocab_size=int(_maybe_get(config, "vocab_size", _maybe_get(text_cfg, "vocab_size", 0))),
        pad_token_id=_maybe_get(config, "pad_token_id", _maybe_get(text_cfg, "pad_token_id")),
        bos_token_id=_maybe_get(config, "bos_token_id", _maybe_get(text_cfg, "bos_token_id")),
        eos_token_id=_maybe_get(config, "eos_token_id", _maybe_get(text_cfg, "eos_token_id")),
        image_token_id=int(_maybe_get(config, "image_token_id", IMAGE_TOKEN_ID)),
        video_token_id=int(_maybe_get(config, "video_token_id", VIDEO_TOKEN_ID)),
        vision_start_token_id=int(_maybe_get(config, "vision_start_token_id", VISION_START_TOKEN_ID)),
        vision_end_token_id=int(_maybe_get(config, "vision_end_token_id", VISION_END_TOKEN_ID)),
        tie_word_embeddings=bool(_maybe_get(config, "tie_word_embeddings", False)),
        text=text_spec,
        vision=vision_spec,
    )


def get_rope_index(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D RoPE indices used by the official finetuning codebase.

    This function mirrors ``qwenvl.data.rope2d.get_rope_index`` so we can keep
    the preprocessing logic close to the rest of the LMPO codebase.
    """

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas: list[int] = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, seq_input_ids in enumerate(total_input_ids):  # per-sample pass
            seq_input_ids = seq_input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(seq_input_ids == VISION_START_TOKEN_ID).squeeze(1)
            vision_tokens = seq_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == IMAGE_TOKEN_ID).sum()
            video_nums = (vision_tokens == VIDEO_TOKEN_ID).sum()
            input_tokens = seq_input_ids.tolist()
            llm_pos_ids_list: List[torch.Tensor] = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(int(image_nums + video_nums)):  # walk segments
                if IMAGE_TOKEN_ID in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(IMAGE_TOKEN_ID, st)
                else:
                    ed_image = len(input_tokens) + 1
                if VIDEO_TOKEN_ID in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(VIDEO_TOKEN_ID, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = (
                    torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(int(llm_positions.max() + 1 - len(total_input_ids[i])))

        mrope_position_deltas_tensor = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas_tensor
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas_tensor = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas_tensor = torch.zeros(
                [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
            )

        return position_ids, mrope_position_deltas_tensor

@dataclass
class Qwen3VLResources:
    """Container with the core Qwen3-VL artefacts."""

    processor: AutoProcessor
    model: Qwen3VLMoeForConditionalGeneration
    device: torch.device


def _resolve_device(requested: str | torch.device | None) -> torch.device:
    if isinstance(requested, torch.device):  # already resolved
        return requested
    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_qwen3_vl(
    model_id: str,
    *,
    device: str | torch.device | None = "auto",
    torch_dtype: torch.dtype | str = "auto",
    attn_implementation: str | None = None,
    device_map: str | dict | None = "auto",
) -> Qwen3VLResources:
    """Load a Qwen3-VL checkpoint using Hugging Face Transformers.

    # thin convenience around from_pretrained
    """

    # Rely on HF's automatic device_map to shard large checkpoints when possible.
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    resolved_device = _resolve_device(device)  # cpu/cuda/mps
    model.to(resolved_device)

    # Left padding keeps newly generated tokens contiguous at the end of the tensor.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return Qwen3VLResources(processor=processor, model=model, device=resolved_device)


def _clean_image_inputs(images: Sequence[str | Path]) -> List[str]:
    cleaned: List[str] = []
    for item in images:
        if isinstance(item, Path):
            cleaned.append(str(item))
        else:
            cleaned.append(str(item))
    return cleaned  # normalised paths


class Qwen3VLWrapper:
    """High-level convenience wrapper for Qwen3-VL."""

    def __init__(self, resources: Qwen3VLResources):
        self.processor = resources.processor
        self.model = resources.model
        self.device = resources.device
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.image_patch_size = getattr(self.model.config, "image_patch_size", 16)

    @property
    def tokenizer(self):  # typed alias
        return self.processor.tokenizer

    def describe_structure(self) -> Qwen3VLStructure:
        """Return a dataclass summarising the HF config."""

        return describe_qwen3_vl(self.model)  # one call summary

    def compute_rope_indices(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        variant: str = "qwen3",
        spatial_merge_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D RoPE indices from tokenised processor outputs."""

        if "input_ids" not in inputs:
            raise KeyError("inputs must contain `input_ids`")

        rope_fn_map = {
            "qwen3": get_rope_index,
        }
        key = variant.lower()
        if key not in rope_fn_map:
            raise ValueError(f"Unsupported RoPE variant '{variant}'. Expected one of {sorted(rope_fn_map)}")
        rope_fn = rope_fn_map[key]

        image_grid_thw = inputs.get("image_grid_thw")
        video_grid_thw = inputs.get("video_grid_thw")
        second_per_grid_ts = inputs.get("second_per_grid_ts")
        attention_mask = inputs.get("attention_mask")
        merge = spatial_merge_size  # default to config if None
        if merge is None:
            merge = getattr(self.model.config.vision_config, "spatial_merge_size", 2)

        return rope_fn(
            spatial_merge_size=merge,
            input_ids=inputs["input_ids"],
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )

    def _extract_content_tokens(self, seq: torch.Tensor, prompt_prefix_len: int) -> list[int]:
        """Extract content tokens: strip left padding prefix and stop at EOS/pad."""
        generated = seq[prompt_prefix_len:]
        mask = (generated != self.pad_token_id) & (generated != self.eos_token_id)
        if not mask.any():
            return []
        end_idx = (~mask).long().argmax() if (~mask).any() else len(generated)
        return generated[:end_idx].tolist()

    # ------------------------------------------------------------------
    # Prompt/message helpers
    # ------------------------------------------------------------------
    def build_messages(
        self,
        question: str,
        image_paths: Sequence[str | Path],
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create a Qwen chat message payload for a question + images."""

        content: list[dict[str, Any]] = []
        for image in _clean_image_inputs(image_paths):
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": question})

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})
        return messages

    def prepare_inputs(
        self,
        batched_messages: Sequence[Sequence[dict[str, Any]]],
    ) -> tuple[dict[str, torch.Tensor], list[int], list[list[int]]]:
        """Tokenise text + preprocess images for a batch of conversations."""

        # ``apply_chat_template`` expects a list of conversations. ``tokenize=False`` lets us
        # keep the strings for the processor while still tracking input lengths later via
        # the produced attention mask.
        prompt_texts = self.processor.apply_chat_template(
            batched_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]

        images, videos, video_kwargs = process_vision_info(
            batched_messages,
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        video_list = None
        if videos is not None:
            video_list, metadata_pairs = zip(*videos)
            video_list = list(video_list)
            video_metadatas = [meta for meta in metadata_pairs]

        processor_kwargs: dict[str, Any] = {
            "text": prompt_texts,
            "images": images,
            "videos": video_list,
            "video_metadata": video_metadatas,
            "return_tensors": "pt",
            "padding": True,
        }
        processor_kwargs.update(video_kwargs)

        inputs = self.processor(**processor_kwargs)
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # Record prompt lengths and the non-padded prompt ids for later slicing/logging.
        attention_mask = inputs["attention_mask"]
        prompt_lengths = attention_mask.sum(dim=-1).tolist()
        input_ids = inputs["input_ids"]
        prompt_token_ids = [row[-length:].tolist() for row, length in zip(input_ids, prompt_lengths)]
        return inputs, prompt_lengths, prompt_token_ids

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        questions: Sequence[str],
        image_paths: Sequence[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float | None = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        return_logprobs: bool = False,
        return_prompt: bool = False,
        system_prompt: str | None = None,
    ) -> tuple[list[list[int]], torch.Tensor | None] | tuple[list[list[int]], torch.Tensor | None, list[list[int]]]:
        if len(questions) != len(image_paths):
            raise ValueError("questions and image_paths must have identical length")

        batched_messages = [
            self.build_messages(question, [image_path], system_prompt=system_prompt)
            for question, image_path in zip(questions, image_paths)
        ]
        inputs, _, prompt_token_ids = self.prepare_inputs(batched_messages)
        prompt_prefix_len = inputs["input_ids"].shape[1]

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "temperature": temperature,
            "return_dict_in_generate": return_logprobs,
            "output_scores": return_logprobs,
        }
        if top_p is not None:
            generation_kwargs["top_p"] = top_p

        outputs = self.model.generate(**inputs, **generation_kwargs)

        if return_logprobs:
            sequences = outputs.sequences
            scores = outputs.scores
        else:
            sequences = outputs
            scores = None

        generated_token_lists = [
            self._extract_content_tokens(seq, prompt_prefix_len) for seq in sequences
        ]

        logprob_tensor: torch.Tensor | None = None
        if scores is not None and scores:
            batch_size = sequences.shape[0]
            max_len = max((len(tokens) for tokens in generated_token_lists), default=0)
            logprob_tensor = torch.zeros(batch_size, max_len, device=self.device)
            for step_idx, score in enumerate(scores):
                if step_idx >= max_len:
                    break
                logprobs = torch.log_softmax(score.float(), dim=-1)
                for batch_idx, tokens in enumerate(generated_token_lists):
                    if step_idx < len(tokens):
                        token_id = tokens[step_idx]
                        logprob_tensor[batch_idx, step_idx] = logprobs[batch_idx, token_id]

        if return_prompt:
            return generated_token_lists, logprob_tensor, prompt_token_ids

        return generated_token_lists, logprob_tensor

    # ------------------------------------------------------------------
    # Logprob recomputation (for PPO updates)
    # ------------------------------------------------------------------
    def compute_token_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        **additional_inputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-token logprobs and entropies for provided sequences."""

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        model_inputs.update(additional_inputs)

        outputs = self.model(**model_inputs)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
        return token_logprobs, entropy


# ----------------------------------------------------------------------
# Supervised finetuning helpers migrated from ``core/vlm``.
# ----------------------------------------------------------------------


def _rank0_log(data_args: VLMDatasetArguments, message: str) -> None:
    if getattr(data_args, "local_rank", 0) == 0:
        logger.info(message)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _make_abs_paths(base: Path, files: str) -> str:
    return str((base / files).resolve())


def update_processor_pixels(processor: transformers.AutoProcessor, data_args: VLMDatasetArguments):
    """Mutate the underlying HF processor with dataset-specific vision bounds."""

    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        _rank0_log(data_args, "Adjusting image processor pixel ranges")
        if hasattr(ip, "min_pixels"):
            ip.min_pixels = data_args.min_pixels
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = data_args.max_pixels
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = data_args.min_pixels
            ip.size["longest_edge"] = data_args.max_pixels

    vp = getattr(processor, "video_processor", None)
    if vp is not None:
        _rank0_log(data_args, "Adjusting video processor pixel/frame ranges")
        if hasattr(vp, "min_pixels"):
            vp.min_pixels = data_args.video_min_pixels
        if hasattr(vp, "max_pixels"):
            vp.max_pixels = data_args.video_max_pixels
        if hasattr(vp, "min_frames"):
            vp.min_frames = data_args.video_min_frames
        if hasattr(vp, "max_frames"):
            vp.max_frames = data_args.video_max_frames
        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]
    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    image_pool = [{"type": "image", "image": _make_abs_paths(base_path, img)} for img in images]
    video_pool = [{"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos]

    messages: List[Dict[str, Any]] = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]
        if role == "user":
            content: List[Dict[str, Any]] = []
            text_parts = re.split(r"(<image>|<video>)", text)
            for segment in text_parts:
                if segment == DEFAULT_IMAGE_TOKEN:
                    if not image_pool:
                        raise ValueError("<image> placeholder has no matching asset")
                    content.append(image_pool.pop(0))
                elif segment == DEFAULT_VIDEO_TOKEN:
                    if not video_pool:
                        raise ValueError("<video> placeholder has no matching asset")
                    content.append(video_pool.pop(0))
                elif segment.strip():
                    content.append({"type": "text", "text": segment.strip()})
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) not referenced by the prompt")
    if video_pool:
        raise ValueError(f"{len(video_pool)} video(s) not referenced by the prompt")
    return messages


def preprocess_qwen_visual(
    sources: Sequence[Dict[str, Any]],
    processor: transformers.AutoProcessor,
) -> Dict[str, Any]:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", "")) if source.get("data_path") else Path("")
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    result: Dict[str, Any] = dict(full_result)
    input_ids = result.get("input_ids")
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        result["input_ids"] = input_ids

    labels = torch.full_like(result["input_ids"], IGNORE_INDEX)
    flat_ids = result["input_ids"][0].tolist()
    length = len(flat_ids)
    pos = 0
    while pos < length:
        if flat_ids[pos] == 77091:  # <answer>
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < length and flat_ids[ans_end] != 151645:
                ans_end += 1
            if ans_end < length:
                labels[0, ans_start : ans_end + 2] = result["input_ids"][0, ans_start : ans_end + 2]
                pos = ans_end
        pos += 1

    result["labels"] = labels
    return result


class LazySupervisedDataset(Dataset):
    """Dataset for multimodal supervised finetuning."""

    def __init__(
        self,
        processor: transformers.AutoProcessor,
        data_args: VLMDatasetArguments,
        dataset_specs: Optional[Sequence[DatasetSpec]] = None,
    ) -> None:
        super().__init__()
        specs = list(dataset_specs or data_args.parsed_datasets())
        if not specs:
            raise ValueError("No dataset specifications provided for supervised finetuning")

        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = self.processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)

        if data_args.model_type != "qwen3vl":
            raise ValueError(f"Unsupported model_type '{data_args.model_type}'. Only 'qwen3vl' is handled.")
        self.get_rope_index = get_rope_index

        self.list_data_dict: list[Any] = []
        for spec in specs:
            annotations_path = Path(spec.annotation_path)
            if not annotations_path.exists():
                raise FileNotFoundError(f"Dataset annotation file not found: {annotations_path}")
            if annotations_path.suffix == ".jsonl":
                annotations = read_jsonl(str(annotations_path))
            else:
                with open(annotations_path, "r", encoding="utf-8") as handle:
                    annotations = json.load(handle)

            if spec.sampling_rate < 1.0:
                take = max(1, int(len(annotations) * spec.sampling_rate))
                annotations = random.sample(annotations, take)

            base_path = spec.data_path or str(annotations_path.parent)
            for ann in annotations:
                if isinstance(ann, list):
                    for sub in ann:
                        sub.setdefault("data_path", base_path)
                else:
                    ann.setdefault("data_path", base_path)
            self.list_data_dict.extend(annotations)

        random.shuffle(self.list_data_dict)

        self.item_fn = self._get_packed_item if data_args.data_packing else self._get_item

    def __len__(self) -> int:
        return len(self.list_data_dict)

    @property
    def lengths(self) -> List[int]:
        length_list: List[int] = []
        for sample in self.list_data_dict:
            img_tokens = 128 if isinstance(sample, dict) and "image" in sample else 0
            text_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            length_list.append(text_len + img_tokens)
        return length_list

    @property
    def modality_lengths(self) -> List[int]:
        length_list: List[int] = []
        for sample in self.list_data_dict:
            text_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            has_media = ("image" in sample) or ("video" in sample)
            length_list.append(text_len if has_media else -text_len)
        return length_list

    @property
    def pre_calculated_length(self) -> np.ndarray:
        if self.list_data_dict and "num_tokens" in self.list_data_dict[0]:
            return np.array([sample["num_tokens"] for sample in self.list_data_dict])
        return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        attempts = 3
        for _ in range(attempts):
            try:
                sources = self.list_data_dict[index]
                if isinstance(sources, dict):
                    sources = [sources]
                return self.item_fn(sources)
            except Exception as exc:  # pragma: no cover - dataset robustness
                logger.warning("Failed to read sample %s (%s). Retrying...", index, exc)
                time.sleep(1)
        # last attempt without catching
        sources = self.list_data_dict[index]
        if isinstance(sources, dict):
            sources = [sources]
        return self.item_fn(sources)

    def _get_item(self, sources: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(sources, self.processor)
        seq_len = data_dict["input_ids"][0].size(0)

        image_grid_thw = data_dict.get("image_grid_thw")
        if image_grid_thw is not None and not isinstance(image_grid_thw, Sequence):
            image_grid_thw = [image_grid_thw]
        video_grid_thw = data_dict.get("video_grid_thw")
        if video_grid_thw is not None and not isinstance(video_grid_thw, Sequence):
            video_grid_thw = [video_grid_thw]

        seconds_per_grid: Optional[List[float]] = None
        if video_grid_thw:
            vp = getattr(self.processor, "video_processor", None)
            if vp is not None and getattr(vp, "fps", None):
                temporal_patch = getattr(vp, "temporal_patch_size", 1)
                fps = getattr(vp, "fps", 1.0)
                seconds = float(temporal_patch) / float(fps)
                seconds_per_grid = [seconds] * len(video_grid_thw)

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(image_grid_thw, dim=0) if image_grid_thw else None,
            video_grid_thw=torch.cat(video_grid_thw, dim=0) if video_grid_thw else None,
            second_per_grid_ts=seconds_per_grid,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        return data_dict

    def _get_packed_item(self, sources: Sequence[Any]) -> Dict[str, torch.Tensor]:
        if isinstance(sources, dict):
            sources = [sources]

        batch_items: List[Dict[str, torch.Tensor]] = []
        for source in sources:
            if isinstance(source, dict):
                batch_items.append(self._get_item([source]))
            elif isinstance(source, Sequence):
                batch_items.append(self._get_item(source))
            else:
                raise TypeError("Unexpected packed source format")

        input_ids = torch.cat([item["input_ids"] for item in batch_items], dim=1)
        labels = torch.cat([item["labels"] for item in batch_items], dim=1)
        position_ids = torch.cat([item["position_ids"] for item in batch_items], dim=2)

        attention_mask = [
            item["attention_mask"][0]
            for item in batch_items
            if isinstance(item.get("attention_mask"), Sequence)
        ]

        new_dict: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask if attention_mask else None,
        }

        if any("pixel_values" in item for item in batch_items):
            new_dict["pixel_values"] = torch.cat(
                [item["pixel_values"] for item in batch_items if "pixel_values" in item], dim=0
            )
            new_dict["image_grid_thw"] = torch.cat(
                [item["image_grid_thw"] for item in batch_items if "image_grid_thw" in item], dim=0
            )

        if any("pixel_values_videos" in item for item in batch_items):
            new_dict["pixel_values_videos"] = torch.cat(
                [item["pixel_values_videos"] for item in batch_items if "pixel_values_videos" in item], dim=0
            )
            new_dict["video_grid_thw"] = torch.cat(
                [item["video_grid_thw"] for item in batch_items if "video_grid_thw" in item], dim=0
            )

        return new_dict


def pad_and_cat(tensor_list: Sequence[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded = [F.pad(tensor, (0, max_length - tensor.shape[2]), value=1) for tensor in tensor_list]
    return torch.cat(padded, dim=1)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised finetuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = (
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "position_ids": position_ids,
        }

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance], dim=0
            )

        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance]
        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat(
                [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance], dim=0
            )

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate packed multimodal examples."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = (
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        flat_attention = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if instance.get("attention_mask") is not None
                )
            )
        )
        seq_lens = torch.tensor([0] + flat_attention, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": cumsum_seq_lens,
            "position_ids": position_ids,
        }

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance], dim=0
            )

        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance]
        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat(
                [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance], dim=0
            )

        return batch


def make_supervised_data_module(
    wrapper: Qwen3VLWrapper,
    data_args: VLMDatasetArguments,
    dataset_specs: Optional[Sequence[DatasetSpec]] = None,
) -> Dict[str, Any]:
    dataset = LazySupervisedDataset(wrapper.processor, data_args=data_args, dataset_specs=dataset_specs)
    if data_args.data_flatten or data_args.data_packing:
        collator = FlattenedDataCollatorForSupervisedDataset(wrapper.processor.tokenizer)
    else:
        collator = DataCollatorForSupervisedDataset(wrapper.processor.tokenizer)
    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": collator,
    }


def configure_trainable_modules(model: transformers.PreTrainedModel, model_args: VLMModelArguments) -> None:
    """Toggle which submodules participate in finetuning."""

    if hasattr(model, "visual"):
        for _, param in model.visual.named_parameters():
            param.requires_grad = model_args.tune_mm_vision
    if hasattr(model, "visual") and hasattr(model.visual, "merger"):
        for _, param in model.visual.merger.named_parameters():
            param.requires_grad = model_args.tune_mm_mlp

    if hasattr(model, "language_model"):
        for _, param in model.language_model.named_parameters():
            param.requires_grad = model_args.tune_mm_llm
    if hasattr(model, "lm_head"):
        model.lm_head.requires_grad = model_args.tune_mm_llm


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str) -> None:
    """Collect the state dict and dump to disk in a deepspeed-aware manner."""

    if trainer.deepspeed:  # pragma: no cover - requires distributed runtime
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def log_trainable_parameter_summary(model: torch.nn.Module) -> None:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    percent = (trainable / total) * 100 if total else 0.0
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        percent,
    )


class VisionLanguageTrainer(Trainer):
    """Thin Trainer subclass that supports multimodal LR heads."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        args = self.args
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        projector_params = {
            name for name, _ in opt_model.named_parameters() if "merger" in name
        }
        vision_params = {
            name for name, _ in opt_model.named_parameters() if "visual" in name
        }

        optimizer_grouped_parameters: list[dict[str, Any]] = []

        def _collect(condition):
            return [
                param
                for name, param in opt_model.named_parameters()
                if condition(name, param)
            ]

        # Core model decay
        optimizer_grouped_parameters.append(
            {
                "params": _collect(
                    lambda n, p: n in decay_parameters
                    and n not in projector_params
                    and n not in vision_params
                    and p.requires_grad
                ),
                "weight_decay": args.weight_decay,
            }
        )
        # Core model no decay
        optimizer_grouped_parameters.append(
            {
                "params": _collect(
                    lambda n, p: n not in decay_parameters
                    and n not in projector_params
                    and n not in vision_params
                    and p.requires_grad
                ),
                "weight_decay": 0.0,
            }
        )

        if args.vision_tower_lr:
            optimizer_grouped_parameters.append(
                {
                    "params": _collect(
                        lambda n, p: n in decay_parameters
                        and n in vision_params
                        and n not in projector_params
                        and p.requires_grad
                    ),
                    "weight_decay": args.weight_decay,
                    "lr": args.vision_tower_lr,
                }
            )
            optimizer_grouped_parameters.append(
                {
                    "params": _collect(
                        lambda n, p: n not in decay_parameters
                        and n in vision_params
                        and n not in projector_params
                        and p.requires_grad
                    ),
                    "weight_decay": 0.0,
                    "lr": args.vision_tower_lr,
                }
            )

        if args.mm_projector_lr:
            optimizer_grouped_parameters.append(
                {
                    "params": _collect(
                        lambda n, p: n in decay_parameters
                        and n in projector_params
                        and p.requires_grad
                    ),
                    "weight_decay": args.weight_decay,
                    "lr": args.mm_projector_lr,
                }
            )
            optimizer_grouped_parameters.append(
                {
                    "params": _collect(
                        lambda n, p: n not in decay_parameters
                        and n in projector_params
                        and p.requires_grad
                    ),
                    "weight_decay": 0.0,
                    "lr": args.mm_projector_lr,
                }
            )

        optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if group["params"]]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


__all__ = [
    "IMAGE_TOKEN_ID",
    "VIDEO_TOKEN_ID",
    "VISION_START_TOKEN_ID",
    "VISION_END_TOKEN_ID",
    "TextBackboneSpec",
    "VisionBackboneSpec",
    "Qwen3VLStructure",
    "describe_qwen3_vl",
    "get_rope_index",
    "Qwen3VLResources",
    "Qwen3VLWrapper",
    "load_qwen3_vl",
    "VLMModelArguments",
    "VLMDatasetArguments",
    "VLMTrainingArguments",
    "DatasetSpec",
    "LazySupervisedDataset",
    "DataCollatorForSupervisedDataset",
    "FlattenedDataCollatorForSupervisedDataset",
    "make_supervised_data_module",
    "configure_trainable_modules",
    "safe_save_model_for_hf_trainer",
    "preprocess_qwen_visual",
    "update_processor_pixels",
    "log_trainable_parameter_summary",
    "VisionLanguageTrainer",
]
