"""Benchmark and smoke test for the unified JAX sampler.

This entry point now supports comparing the JAX sampler against a Hugging Face
PyTorch reference model and reports basic latency/throughput statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from vlmrl.core.types import SampleResult, SamplingConfig, VLMInputs
from vlmrl.core.sampling import sample as unified_sample
from vlmrl.models.qwen3vl.model import VisionEmbeddings, create_model_from_ckpt as create_model_qwen3
from vlmrl.utils.vlm import preprocess_image, chat_prompt_with_image, decode_tokens, extract_assistant
from vlmrl.utils.rng import RngSeq


def _resolve_pad_eos(tokenizer, model, pad_id_flag: Optional[int]) -> tuple[int, Optional[int]]:
    pad_id = (
        int(pad_id_flag)
        if pad_id_flag is not None
        else (getattr(tokenizer, "pad_token_id", None) or getattr(model.spec, "pad_token_id", 0) or 0)
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(model.spec, "eos_token_id", None)
    return int(pad_id), (int(eos_id) if eos_id is not None else None)


def _resolve_image_pad_id(tokenizer, ckpt_dir: str) -> int:
    """Resolve the special <image> placeholder token id for Qwen3.

    Tries tokenizer first; falls back to checkpoint config.json.
    """
    import json, os
    try:
        image_pad = getattr(tokenizer, "image_token_id", None)
        if image_pad is not None and int(image_pad) >= 0:
            return int(image_pad)
        special = getattr(tokenizer, "special_tokens_map", None)
        if isinstance(special, dict):
            image_pad = special.get("image_token_id", None)
            if image_pad is not None and int(image_pad) >= 0:
                return int(image_pad)
    except Exception:
        pass
    try:
        with open(os.path.join(ckpt_dir, "config.json"), "r") as f:
            cfg = json.load(f)
        return int(cfg.get("image_token_id", 151655))
    except Exception:
        return 151655


def _ensure_ready(result: SampleResult) -> None:
    """Block until the JAX computation backing `result` has finished."""
    tokens = result.tokens
    if isinstance(tokens, jax.Array):
        jax.block_until_ready(tokens)
    if result.logprobs is not None and isinstance(result.logprobs, jax.Array):
        jax.block_until_ready(result.logprobs)


def _summaries_from_metrics(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate timing metrics collected over multiple runs."""
    if not metrics:
        return {}
    durations = [m["duration"] for m in metrics]
    tokens = [m.get("tokens", 0.0) for m in metrics]
    throughputs = [
        (tok / dur) if dur > 0 else 0.0
        for tok, dur in zip(tokens, durations, strict=False)
    ]
    summary = {
        "runs": len(durations),
        "duration_mean": statistics.mean(durations),
        "duration_std": statistics.pstdev(durations) if len(durations) > 1 else 0.0,
        "tokens_total": sum(tokens),
        "throughput_mean": statistics.mean(throughputs),
        "throughput_std": statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0,
    }
    if len(durations) > 1:
        steady = durations[1:]
        steady_tokens = tokens[1:]
        steady_tp = [
            (tok / dur) if dur > 0 else 0.0
            for tok, dur in zip(steady_tokens, steady, strict=False)
        ]
        summary.update({
            "steady_duration_mean": statistics.mean(steady),
            "steady_throughput_mean": statistics.mean(steady_tp),
        })
    return summary


def _format_summary_table(title: str, summary: Dict[str, float], first_token: Optional[float]) -> str:
    lines = [f"{title}:"]
    if not summary:
        lines.append("  (no data)")
        return "\n".join(lines)
    lines.append(f"  runs: {summary['runs']}")
    lines.append(f"  mean duration: {summary['duration_mean']:.3f}s (std {summary['duration_std']:.3f})")
    lines.append(f"  total tokens: {int(summary['tokens_total'])}")
    lines.append(f"  mean throughput: {summary['throughput_mean']:.2f} tok/s (std {summary['throughput_std']:.2f})")
    if "steady_duration_mean" in summary:
        lines.append(f"  steady-state duration: {summary['steady_duration_mean']:.3f}s")
        lines.append(f"  steady-state throughput: {summary['steady_throughput_mean']:.2f} tok/s")
    if first_token is not None:
        lines.append(f"  first-token latency (1 new token run): {first_token:.3f}s")
    return "\n".join(lines)


def _ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _benchmark_jax(
    model,
    params,
    inputs: Union[VLMInputs, jnp.ndarray],
    cfg: SamplingConfig,
    rng_seq: RngSeq,
    tokenizer,
    runs: int,
) -> tuple[SampleResult, Dict[str, float], Optional[float]]:
    metrics: List[Dict[str, float]] = []
    last_result: Optional[SampleResult] = None
    measured_runs = max(1, int(runs))
    warmup_runs = 1 if measured_runs > 0 else 0

    # Prefill compilation cache with an untimed warm-up pass
    for idx in range(warmup_runs + measured_runs):
        key = rng_seq.next()
        start = time.perf_counter()
        result = unified_sample(
            model,
            params,
            inputs,
            cfg,
            key,
            tokenizer=tokenizer,
            return_logprobs=False,
        )
        _ensure_ready(result)
        elapsed = time.perf_counter() - start
        if idx >= warmup_runs:
            tokens_generated = int(result.tokens.shape[1]) if result.tokens.ndim == 2 else 0
            metrics.append({"duration": elapsed, "tokens": tokens_generated})
            last_result = result
        else:
            # Warm-up result is only used to trigger compilation
            last_result = result

    result_one: Optional[SampleResult] = None
    first_token_latency: Optional[float] = None
    try:
        cfg_one = replace(cfg, max_new_tokens=1)
        key = rng_seq.next()
        result_one = unified_sample(
            model,
            params,
            inputs,
            cfg_one,
            key,
            tokenizer=tokenizer,
            return_logprobs=False,
        )
        _ensure_ready(result_one)  # warm-up compile
        # Timed single-token decode
        key = rng_seq.next()
        start = time.perf_counter()
        result_one = unified_sample(
            model,
            params,
            inputs,
            cfg_one,
            key,
            tokenizer=tokenizer,
            return_logprobs=False,
        )
        _ensure_ready(result_one)
        elapsed = time.perf_counter() - start
        generated = int(result_one.tokens.shape[1]) if result_one.tokens.ndim == 2 else 0
        if generated > 0:
            first_token_latency = elapsed
    except Exception:
        first_token_latency = None

    final_result = last_result or result_one
    if final_result is None:
        raise RuntimeError("JAX sampling did not produce a result.")
    return final_result, _summaries_from_metrics(metrics), first_token_latency


def _benchmark_hf_text(
    model_name: str,
    prompt_text: str,
    tokenizer_kwargs: Dict[str, Any],
    max_new_tokens: int,
    runs: int,
    *,
    dtype: str = "auto",
    messages: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, float]]:
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer as HFAutoTokenizer,
            AutoProcessor,
        )
    except ImportError:
        return None

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]

    hf_tokenizer = None
    tokenizer_error: Optional[Exception] = None
    try:
        hf_tokenizer = HFAutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - best effort fallback
        tokenizer_error = exc

    processor: Optional[Any] = None
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        processor = None

    model_dtype = dtype
    if isinstance(dtype, str) and dtype != "auto":
        model_dtype = getattr(torch, dtype, dtype)

    # Try VLM class first before falling back to CausalLM
    model = None
    if Qwen3VLForConditionalGeneration is not None:
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=model_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except (ValueError, OSError):
            model = None

    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=model_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                print(f"Model {model_name} is not compatible with AutoModelForCausalLM or VLM classes")
                return None
            raise

    # Prepare encoded inputs either from processor chat template or tokenizer fallback
    batch_inputs = None
    input_length: Optional[int] = None
    if processor is not None and hasattr(processor, "apply_chat_template") and messages:
        try:
            batch_inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch_inputs = batch_inputs.to(model.device)
            batch_inputs = {k: v for k, v in batch_inputs.items()}
            input_length = int(batch_inputs["input_ids"].shape[1])
        except Exception:
            batch_inputs = None
            input_length = None

    if batch_inputs is None:
        if hf_tokenizer is None:
            raise RuntimeError(
                f"Tokenizer for model {model_name} failed to load: {tokenizer_error!s}"
                if tokenizer_error is not None
                else f"Tokenizer for model {model_name} is unavailable."
            )
        batch_inputs = hf_tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
            **tokenizer_kwargs,
        )
        batch_inputs = batch_inputs.to(model.device)
        batch_inputs = {k: v for k, v in batch_inputs.items()}
        input_length = int(batch_inputs["input_ids"].shape[1])

    metrics: List[Dict[str, float]] = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=max_new_tokens,
        )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - start
        generated = int(outputs[0].shape[0] - input_length)
        metrics.append({"duration": elapsed, "tokens": generated})

    # First token latency
    first_token_latency: Optional[float] = None
    try:
        start = time.perf_counter()
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=1,
        )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - start
        generated = int(outputs[0].shape[0] - input_length)
        if generated > 0:
            first_token_latency = elapsed
    except Exception:
        first_token_latency = None

    summary = _summaries_from_metrics(metrics)
    if first_token_latency is not None:
        summary["first_token_latency"] = first_token_latency
    return summary


def _benchmark_hf_vision(
    model_name: str,
    image_path: str,
    prompt_text: str,
    max_new_tokens: int,
    runs: int,
    *,
    dtype: str = "auto",
) -> Optional[Dict[str, float]]:
    """Benchmark HuggingFace VLM on a multimodal image+text task."""
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except ImportError:
        return None

    model_dtype = dtype
    if isinstance(dtype, str) and dtype != "auto":
        model_dtype = getattr(torch, dtype, dtype)

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except (ValueError, OSError) as e:
        print(f"Failed to load VLM model {model_name}: {e}")
        return None

    # Format as multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Apply chat template for multimodal input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    metrics: List[Dict[str, float]] = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - start
        generated = int(generated_ids[0].shape[0] - inputs["input_ids"].shape[1])
        metrics.append({"duration": elapsed, "tokens": generated})

    # First token latency
    first_token_latency: Optional[float] = None
    try:
        start = time.perf_counter()
        generated_ids = model.generate(**inputs, max_new_tokens=1)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - start
        generated = int(generated_ids[0].shape[0] - inputs["input_ids"].shape[1])
        if generated > 0:
            first_token_latency = elapsed
    except Exception:
        first_token_latency = None

    summary = _summaries_from_metrics(metrics)
    if first_token_latency is not None:
        summary["first_token_latency"] = first_token_latency
    return summary


def _save_chart(plot_path: str, measurements: Dict[str, Dict[str, float]]) -> None:
    """Persist a bar chart of mean throughput using matplotlib (if available)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        with open(plot_path, "w", encoding="utf-8") as fh:
            json.dump(measurements, fh, indent=2)
        return

    labels = []
    values = []
    for name, stats in measurements.items():
        labels.append(name)
        values.append(stats.get("throughput_mean", 0.0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#457b9d", "#e63946"])
    ax.set_ylabel("tokens / second")
    ax.set_title("Sampler Throughput")
    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark JAX sampler against Hugging Face reference.")
    parser.add_argument("--ckpt_dir", required=True, type=str, help="Checkpoint directory produced by hf_to_jax.")
    parser.add_argument("--prompt", type=str, default=None, help="User prompt. Required for text or vision runs.")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for VLM sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad_id", type=int, default=None)
    parser.add_argument("--benchmark_runs", type=int, default=3, help="Number of repeated runs when timing samplers.")
    parser.add_argument("--compare_hf", action="store_true", help="Load Hugging Face PyTorch model for reference benchmarking.")
    parser.add_argument("--hf_model_name", type=str, default="Qwen/Qwen3-4B", help="Hugging Face model id to benchmark against.")
    parser.add_argument("--hf_dtype", type=str, default="auto", help="Torch dtype for HF model (e.g. float16, bfloat16, auto).")
    parser.add_argument("--plot_path", type=str, default=None, help="Optional path to save throughput comparison chart (json fallback if matplotlib missing).")
    parser.add_argument("--dump_metrics", type=str, default=None, help="Optional path to persist raw metrics as JSON.")
    args = parser.parse_args()

    # Strictly load Qwen3-VL checkpoints
    model, params = create_model_qwen3(args.ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=False)
    pad_id, eos_id = _resolve_pad_eos(tokenizer, model, args.pad_id)

    cfg = SamplingConfig(
        temperature=float(args.temperature),
        top_p=(float(args.top_p) if 0.0 < float(args.top_p) < 1.0 else None),
        top_k=(int(args.top_k) if args.top_k is not None and int(args.top_k) > 0 else None),
        eos_id=eos_id,
        pad_id=int(pad_id),
        max_new_tokens=int(args.max_new_tokens),
    )
    rng = RngSeq(int(args.seed))
    measurements: Dict[str, Dict[str, float]] = {}

    if args.image is not None:
        if not args.prompt:
            raise SystemExit("Provide --prompt when sampling with an image.")
        if model.spec.vision is None:
            raise SystemExit("This checkpoint has no vision backbone configured.")

        vision_spec = model.spec.vision
        pixel_values, grid_thw = preprocess_image(
            args.image,
            patch_size=vision_spec.patch_size,
            spatial_merge_size=vision_spec.spatial_merge_size,
            temporal_patch_size=vision_spec.temporal_patch_size,
        )
        vision_embeds = model.apply({"params": params}, pixel_values, grid_thw, method=model.encode_vision)
        num_vision_tokens = (
            int(vision_embeds.tokens.shape[0]) if isinstance(vision_embeds, VisionEmbeddings) else int(vision_embeds.shape[0])
        )
        prompt_text = chat_prompt_with_image(num_vision_tokens, args.prompt)
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

        image_pad_id = _resolve_image_pad_id(tokenizer, args.ckpt_dir)
        inputs = VLMInputs(prompt_tokens=prompt_tokens, vision=vision_embeds, grid_thw=grid_thw, image_pad_id=image_pad_id)
        result, summary, first_latency = _benchmark_jax(
            model,
            params,
            inputs,
            cfg,
            rng,
            tokenizer,
            runs=int(args.benchmark_runs),
        )
        measurements["jax"] = {
            **summary,
            "first_token_latency": first_latency,
        }
        # Prefer extracting assistant span from full text; fall back to decoding only new tokens
        new_tokens = result.tokens[0].tolist()
        full_ids = input_ids + new_tokens
        full_text = tokenizer.decode(full_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        answer = extract_assistant(full_text) or decode_tokens(tokenizer, new_tokens)
        print(answer.strip())

        if summary:
            print(_format_summary_table("JAX sampler metrics", summary, first_latency))

        if args.compare_hf:
            hf_summary = _benchmark_hf_vision(
                args.hf_model_name,
                args.image,
                args.prompt,
                int(args.max_new_tokens),
                int(args.benchmark_runs),
                dtype=args.hf_dtype,
            )
            if hf_summary is None:
                print("Failed to load HuggingFace VLM for multimodal comparison.")
            else:
                measurements["hf"] = hf_summary
                print(
                    _format_summary_table(
                        f"Hugging Face ({args.hf_model_name}) metrics",
                        hf_summary,
                        hf_summary.get("first_token_latency", None),
                    )
                )

        if args.dump_metrics:
            _ensure_parent_dir(args.dump_metrics)
            with open(args.dump_metrics, "w", encoding="utf-8") as fh:
                json.dump(measurements, fh, indent=2)
        if args.plot_path and measurements:
            _ensure_parent_dir(args.plot_path)
            _save_chart(args.plot_path, measurements)

        return

    # Text-only sampling
    if not args.prompt:
        raise SystemExit("Provide --prompt for text-only sampling.")
    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    # Run HF comparison first if requested
    if args.compare_hf:
        hf_summary = _benchmark_hf_text(
            args.hf_model_name,
            prompt_text,
            tokenizer_kwargs={"return_attention_mask": True},
            max_new_tokens=int(args.max_new_tokens),
            runs=int(args.benchmark_runs),
            dtype=args.hf_dtype,
            messages=messages,
        )
        if hf_summary is None:
            print("Failed to import torch / transformers for HF comparison.")
        else:
            measurements["hf"] = hf_summary

    # Then run JAX model
    result, summary, first_latency = _benchmark_jax(
        model,
        params,
        prompt_tokens,
        cfg,
        rng,
        tokenizer,
        runs=int(args.benchmark_runs),
    )
    measurements["jax"] = {
        **summary,
        "first_token_latency": first_latency,
    }

    answer = result.texts[0] if result.texts else ""
    print(answer.strip())
    if summary:
        print(_format_summary_table("JAX sampler metrics", summary, first_latency))
    if args.compare_hf and "hf" in measurements:
        hf_summary = measurements["hf"]
        print(
            _format_summary_table(
                f"Hugging Face ({args.hf_model_name}) metrics",
                hf_summary,
                hf_summary.get("first_token_latency", None),
            )
        )

    if args.dump_metrics:
        _ensure_parent_dir(args.dump_metrics)
        with open(args.dump_metrics, "w", encoding="utf-8") as fh:
            json.dump(measurements, fh, indent=2)
    if args.plot_path and measurements:
        _ensure_parent_dir(args.plot_path)
        _save_chart(args.plot_path, measurements)


if __name__ == "__main__":
    main()
