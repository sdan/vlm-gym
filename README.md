<img width="1027" height="566" alt="Screenshot 2025-10-09 at 12 10 39 PM" src="https://github.com/user-attachments/assets/43d593ed-3426-4532-8462-f3108dcf4f33" />

# vlmrl

A reinforcement learning framework for vision-language models, written in JAX.

**Core components:**
- `models/qwen25vl` — Qwen2.5-VL with mRoPE, KV cache, grouped-query attention
- `core/sampling.py` — Inference
- `core/grpo.py` — Training (GRPO)
- `core/eval.py` — Evaluation
- `envs/base.py` — Vision environments for captioning, multimodal reasoning, etc.

---

## Quickstart

**Install**
```bash
uv sync
```

**Convert HF → JAX** (defaults to Qwen/Qwen2.5-VL-7B-Instruct)
```bash
python -m utils.hf_to_jax --model_dir checkpoints/qwen25vl_7b
```

**Sample**
```bash
python -m core.sampling \
  --ckpt_dir checkpoints/qwen25vl_7b \
  --image imgs/f35_takeoff.png \
  --prompt "Describe the image"
```

**Train (GRPO)**
```bash
python core/grpo.py \
  --model_dir=checkpoints/qwen25vl_7b \
  --env_name=vision \
  --groups_per_batch=8 \
  --group_size=1 \
  --lr=5e-7 \
  --total_steps=10000 \
  --wandb_project=vlm-rl
```

**Eval**
```bash
python core/eval.py \
  --model_dir checkpoints/qwen25vl_7b \
  --env_name=vision \
  --num_generation_tokens=128 \
  --inference_batch_per_device=1 \
  --vlm_max_pixels=1048576 \
  --top_k=5
```

---

## Environments

Extend `envs.base.BaseEnv` to add custom vision environments.

**Built-in:**
- `vision` / `vision_caption` — Single-image captioning; reward = keyword hits
- `nlvr2` — Two-image True/False reasoning

---

## Requirements

- Python 3.10+
- Linux, CUDA 12, NVIDIA GPU (~60GB VRAM for 7B)
- JAX 0.6.1 (CUDA 12 build)

---

## References

- **lmpo** — [kvfrans/lmpo](https://github.com/kvfrans/lmpo)
- **Qwen model base** — [jax-ml/jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main/qwen3)
- **NLVR2 dataset** — [HuggingFaceM4/NLVR2](https://huggingface.co/datasets/HuggingFaceM4/NLVR2)

---

## License

See `LICENSE` and `NOTICE`.
