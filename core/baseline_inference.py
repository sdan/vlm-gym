"""Baseline inference utilities for running Hugging Face or vLLM backends.

This module provides light wrappers around publicly released Qwen2.5-VL
checkpoints so that environments such as ``vision`` can be exercised without
going through the JAX training stack.  The helpers are intentionally minimal:
they convert an environment observation into the chat template expected by the
backend, run generation, then hand the produced token ids back to the
environment for scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from PIL import Image

from vlmrl.envs.env_creator import create_env


ImageInput = Union[str, Image.Image]


def _load_image(image: ImageInput) -> Image.Image:
    """Convert a path or PIL image into a normalized RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image}")
    with Image.open(path) as img:
        return img.convert("RGB")


def _shape_images(images: Sequence[Image.Image]) -> Union[Image.Image, List[Image.Image], List[List[Image.Image]]]:
    """Return the structure expected by HF/vLLM processors.

    Hugging Face processors typically accept either a single PIL image, a list
    of PIL images (parallel batching), or a list of lists when multiple images
    belong to the same sample.  We follow the last convention for multi-image
    prompts so that downstream utilities can handle both cases.
    """
    if not images:
        raise ValueError("At least one image must be provided.")
    if len(images) == 1:
        return images[0]
    return [images]  # nest for multi-image single sample


@dataclass
class GenerationResult:
    """Container for backend outputs."""

    text: str
    token_ids: List[int]
    raw: Any | None = None


class HuggingFaceQwen25VLBackend:
    """Thin wrapper around ``AutoModelForVision2Seq`` for Qwen2.5-VL checkpoints."""

    def __init__(
        self,
        model_id: str,
        *,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **model_kwargs: Any,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Transformers with vision support is required for the Hugging Face backend. "
                "Install via `pip install transformers accelerate`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        dtype_arg: Any = "auto"
        if torch_dtype != "auto":
            torch_dtype = torch_dtype.lower()
            if not hasattr(torch, torch_dtype):
                raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
            dtype_arg = getattr(torch, torch_dtype)

        model_kwargs = dict(model_kwargs)  # shallow copy
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        model_kwargs.setdefault("device_map", device_map)
        if dtype_arg != "auto":
            model_kwargs.setdefault("torch_dtype", dtype_arg)
        if load_in_4bit:
            model_kwargs.setdefault("load_in_4bit", True)
        if load_in_8bit:
            model_kwargs.setdefault("load_in_8bit", True)

        self.model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        # Expose device so tensors can be moved consistently
        self._device = next(self.model.parameters()).device

    def generate(
        self,
        *,
        question: str,
        images: Sequence[ImageInput],
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_token: Optional[int] = None,
    ) -> GenerationResult:
        import torch

        pil_images = [_load_image(img) for img in images]
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in pil_images),
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        processor_kwargs: dict[str, Any] = {
            "text": [prompt_text],
            "images": _shape_images(pil_images),
            "return_tensors": "pt",
        }
        inputs = self.processor(**processor_kwargs)
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
        if temperature is not None and temperature > 0:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["temperature"] = 0.0
            gen_kwargs["do_sample"] = False
        if top_p is not None and 0 < top_p < 1:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None and top_k > 0:
            gen_kwargs["top_k"] = int(top_k)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        continuation = generated[:, input_len:]

        if continuation.numel() == 0:
            return GenerationResult(text="", token_ids=[])

        decoded = self.processor.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        token_ids = continuation[0].tolist()
        if stop_token is not None:
            try:
                stop_index = token_ids.index(int(stop_token))
                token_ids = token_ids[: stop_index + 1]
            except ValueError:
                pass
        return GenerationResult(text=decoded.strip(), token_ids=token_ids, raw=generated)


class VLLMQwen25VLBackend:
    """Wrapper around vLLM's multimodal interface for Qwen2.5-VL checkpoints."""

    def __init__(self, model_id: str, **llm_kwargs: Any) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "vLLM is required for the vLLM backend. Install via `pip install vllm`."
            ) from exc

        llm_kwargs = dict(llm_kwargs)
        llm_kwargs.setdefault("trust_remote_code", True)
        self._llm = LLM(model=model_id, **llm_kwargs)

        tokenizer = self._llm.get_tokenizer()
        if tokenizer is None:
            raise RuntimeError("Failed to retrieve tokenizer from vLLM backend.")
        self.tokenizer = tokenizer

        try:
            processor = self._llm.get_processor()
        except AttributeError:
            processor = None
        self._processor = processor

    def generate(
        self,
        *,
        question: str,
        images: Sequence[ImageInput],
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_token: Optional[int] = None,
    ) -> GenerationResult:
        from vllm import SamplingParams
        from vllm.multimodal.utils import load_image

        pil_images = [_load_image(img) for img in images]
        # vLLM expects numpy image arrays via helper
        mm_images = [load_image(image) for image in pil_images]

        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in pil_images),
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=max(float(temperature), 1e-6),
            top_p=top_p,
            top_k=top_k,
        )
        request = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": mm_images},
        }
        outputs = self._llm.generate([request], sampling_params=sampling)
        if not outputs or not outputs[0].outputs:
            return GenerationResult(text="", token_ids=[])

        text = outputs[0].outputs[0].text.strip()
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if stop_token is not None:
            try:
                idx = token_ids.index(int(stop_token))
                token_ids = token_ids[: idx + 1]
            except ValueError:
                pass
        return GenerationResult(text=text, token_ids=token_ids, raw=outputs)


def _extract_question_and_images(obs: Any) -> tuple[str, List[ImageInput]]:
    """Normalize observation into a question and list of images."""
    if hasattr(obs, "question") and hasattr(obs, "image_path"):
        return obs.question, [getattr(obs, "image_path")]
    if hasattr(obs, "statement") and hasattr(obs, "image_left") and hasattr(obs, "image_right"):
        question = f"Look at the two images. {obs.statement}"
        return question, [getattr(obs, "image_left"), getattr(obs, "image_right")]
    raise ValueError(
        "Unsupported observation type for baseline inference. Expected attributes "
        "like `question`/`image_path` or `statement`/`image_left`/`image_right`."
    )


def run_env(
    backend: Any,
    env_name: str,
    *,
    max_tasks: Optional[int] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    log_interval: int = 0,
) -> dict[str, Any]:
    """Roll out an environment using the provided backend."""

    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Backend must expose a `tokenizer` attribute for env creation.")

    env = create_env(env_name, tokenizer)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    total_tasks = env.num_tasks if env.num_tasks != -1 else len(getattr(env, "dataset", []))
    if total_tasks <= 0:
        raise ValueError("Environment does not expose a finite number of tasks.")
    if max_tasks is not None:
        total_tasks = min(total_tasks, int(max_tasks))

    rewards: List[float] = []
    samples: List[dict[str, Any]] = []

    for idx in range(total_tasks):
        state, obs = env.reset(idx)
        question, image_inputs = _extract_question_and_images(obs)
        result = backend.generate(
            question=question,
            images=image_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token=eos_id,
        )
        action_tokens = result.token_ids
        if eos_id is not None and (not action_tokens or action_tokens[-1] != eos_id):
            action_tokens = action_tokens + [int(eos_id)]
        _, _, reward, done, info = env.step(state, action_tokens)
        rewards.append(float(reward))
        samples.append(
            {
                "question": question,
                "response": result.text,
                "reward": float(reward),
                "keywords": getattr(obs, "keywords", ()),
                "meta": info,
            }
        )
        if log_interval > 0 and (idx + 1) % log_interval == 0:
            print(
                f"[{idx + 1}/{total_tasks}] reward={reward:.3f} response={result.text[:80].strip()}"
            )
        if not done:
            raise RuntimeError("Baseline runner currently assumes bandit-style tasks (done=True).")

    avg_reward = sum(rewards) / max(len(rewards), 1)
    return {"average_reward": avg_reward, "rewards": rewards, "samples": samples}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hugging Face or vLLM Qwen2.5-VL backends on built-in environments."
    )
    parser.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        required=True,
        help="Which backend to use for generation.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model identifier or local path.",
    )
    parser.add_argument("--env_name", default="vision", help="Environment name to evaluate.")
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional task cap.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k shortlist.")
    parser.add_argument("--log_interval", type=int, default=0, help="Print progress every N tasks.")
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Device map hint passed to the Hugging Face backend.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        help="Torch dtype name for the Hugging Face backend (e.g. bfloat16, float16, auto).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Optional tensor parallelism for vLLM backend.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="Optional GPU memory utilization hint for vLLM backend.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    if args.backend == "hf":
        backend = HuggingFaceQwen25VLBackend(
            args.model,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
    else:
        vllm_kwargs: dict[str, Any] = {}
        if args.tensor_parallel_size:
            vllm_kwargs["tensor_parallel_size"] = int(args.tensor_parallel_size)
        if args.gpu_memory_utilization:
            vllm_kwargs["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
        backend = VLLMQwen25VLBackend(args.model, **vllm_kwargs)

    result = run_env(
        backend,
        args.env_name,
        max_tasks=args.max_tasks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        log_interval=args.log_interval,
    )
    print(f"Average reward over {len(result['rewards'])} tasks: {result['average_reward']:.4f}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
