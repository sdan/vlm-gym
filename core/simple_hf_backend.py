"""Minimal Hugging Face baseline runner for Qwen2.5-VL environments.

This module keeps the surface area intentionally small: load a vision-language
checkpoint from Hugging Face, turn each environment observation into a chat
prompt, generate a continuation, and feed the tokens back into the environment
for scoring.  It is useful for sanity-checking datasets before launching JAX
training loops.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Sequence

from PIL import Image

from vlmrl.envs.env_creator import create_env


def _to_image(image: Any) -> Image.Image:
    """Return an RGB PIL image from a filesystem path or PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image}")
    with Image.open(path) as pil:
        return pil.convert("RGB")


def _build_prompt(tokenizer, question: str, num_images: int) -> str:
    """Format a chat prompt matching the Qwen2.5-VL template."""
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_hf_model(
    model_id: str,
    *,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    """Load Hugging Face processor/model/tokenizer trio for Qwen2.5-VL."""
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The simple HF backend requires `transformers` and `torch`. "
            "Install via `pip install transformers accelerate torch`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    dtype_arg: Any = "auto"
    if torch_dtype and torch_dtype != "auto":
        torch_dtype = torch_dtype.lower()
        if not hasattr(torch, torch_dtype):
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        dtype_arg = getattr(torch, torch_dtype)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    if dtype_arg != "auto":
        model_kwargs["torch_dtype"] = dtype_arg

    model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, processor, model, device


def generate_response(
    tokenizer,
    processor,
    model,
    device,
    *,
    question: str,
    images: Sequence[Any],
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_token_id: Optional[int] = None,
) -> tuple[List[int], str]:
    """Generate text and token ids for the given multimodal prompt."""
    import torch

    pil_images = [_to_image(img) for img in images]
    prompt = _build_prompt(tokenizer, question, len(pil_images))

    processor_inputs = processor(
        text=[prompt],
        images=pil_images if len(pil_images) == 1 else [pil_images],
        return_tensors="pt",
    )
    processor_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in processor_inputs.items()}

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
        generated = model.generate(**processor_inputs, **gen_kwargs)

    input_len = processor_inputs["input_ids"].shape[-1]
    continuation = generated[:, input_len:]
    token_ids = continuation[0].tolist()
    if stop_token_id is not None:
        try:
            stop_index = token_ids.index(int(stop_token_id))
            token_ids = token_ids[: stop_index + 1]
        except ValueError:
            pass

    text = processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return token_ids, text


def rollout_env(
    model_id: str,
    env_name: str,
    *,
    max_tasks: Optional[int] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> dict[str, Any]:
    """Evaluate the specified environment with a Hugging Face checkpoint."""
    tokenizer, processor, model, device = _load_hf_model(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    env = create_env(env_name, tokenizer)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    total_tasks = env.num_tasks if env.num_tasks != -1 else len(getattr(env, "dataset", []))
    if total_tasks <= 0:
        raise ValueError("Environment does not expose a finite task dataset.")
    if max_tasks is not None:
        total_tasks = min(total_tasks, int(max_tasks))

    rewards: List[float] = []
    transcripts: List[dict[str, Any]] = []
    for idx in range(total_tasks):
        state, obs = env.reset(idx)
        if hasattr(obs, "question") and hasattr(obs, "image_path"):
            question = obs.question
            image_inputs = [obs.image_path]
        elif hasattr(obs, "statement") and hasattr(obs, "image_left") and hasattr(obs, "image_right"):
            question = f"Look at the two images. {obs.statement}"
            image_inputs = [obs.image_left, obs.image_right]
        else:
            raise ValueError("Unsupported observation type for simple HF backend.")

        token_ids, text = generate_response(
            tokenizer,
            processor,
            model,
            device,
            question=question,
            images=image_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token_id=eos_id,
        )
        if eos_id is not None and (not token_ids or token_ids[-1] != eos_id):
            token_ids = token_ids + [int(eos_id)]

        _, _, reward, done, info = env.step(state, token_ids)
        rewards.append(float(reward))
        transcripts.append(
            {
                "task_index": idx,
                "question": question,
                "response": text,
                "reward": float(reward),
                "keywords": getattr(obs, "keywords", ()),
                "meta": info,
            }
        )
        if not done:
            raise RuntimeError("Environment must terminate after a single response.")

    average_reward = sum(rewards) / max(len(rewards), 1)
    return {
        "average_reward": average_reward,
        "rewards": rewards,
        "samples": transcripts,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple Hugging Face VLM baseline.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model identifier or local checkpoint directory.",
    )
    parser.add_argument("--env_name", default="vision", help="Environment to evaluate.")
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional task cap.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Completion length.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling parameter.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k shortlist parameter.")
    parser.add_argument("--torch_dtype", default="bfloat16", help="Torch dtype name (bfloat16, float16, float32, auto).")
    parser.add_argument("--device_map", default="auto", help="Device map passed to `from_pretrained`.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    result = rollout_env(
        args.model,
        args.env_name,
        max_tasks=args.max_tasks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    print(f"Average reward: {result['average_reward']:.4f} over {len(result['rewards'])} tasks.")


if __name__ == "__main__":  # pragma: no cover
    main()

