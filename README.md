Status: Alpha - PPO support only, expect GRPO support and more documentation. Inference runs smoothly, however training runs currently memoryconstrained to low batch sizes with >80gb of VRAM GPUs. Qwen3-VL-4B-Instruct as the default model.

<img width="325" height="240" alt="vlmgym" src="https://github.com/user-attachments/assets/87d7d141-4464-4687-91c0-3a6da82b2749" />

# vlm-gym

A simple reinforcement learning gym for vision-language models, written in JAX. Drop in any environment, any model, and train with PPO.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdan/vlm-gym/blob/master/train_lora.ipynb)

**Core components:**
- `envs/` — Pluggable vision environments (GeoGuessr, NLVR2, captioning)
- `models/` — VLM implementations (Qwen3-VL-4B-Instruct reference)
- `core/train.py` — Trainer runs PPO on the environment
- `core/rollout.py` — Inference engine runs the VLM on the environment
- `core/eval.py` — Evaluation harness runs the VLM on the environment and compares it to the Hugging Face baseline

---

**Install and convert HF(Qwen3-VL-4B-Instruct default model) → JAX**
```bash
uv sync 
uv run python -m utils.hf_to_jax --model_dir checkpoints/qwen3vl_4b
```

## Run a VLM to play GeoGuessr

```bash
uv run python -m core.rollout 
  --model_dir checkpoints/qwen3vl_4b \
  --env_name geospot
```

## Train a VLM to play GeoGuessr

Training uses a hierarchical curriculum that progressively sharpens geolocation accuracy:
- **Stage 1 (0-100 episodes)**: Country-level coarse matching (wide tolerance)
- **Stage 2 (100-300)**: Country refinement (tighter kernels)
- **Stage 3 (300-600)**: Add region signal
- **Stage 4 (600-1000)**: Introduce city-level precision
- **Stage 5 (1000+)**: Full hierarchical task (country + region + city + coords)

Each field (country/region/city/coords) uses geodesic distance with exponential decay kernels. Weights blend progressively to guide learning from coarse → fine localization.

```bash
# Train on OpenStreetView-5M dataset
uv run python core/train.py \
  --model_dir checkpoints/qwen3vl_4b \
  --env_name geospot \
  --lr 5e-7 \
  --total_steps 10000
```
---

**Sample**
```bash
uv run python -m core.rollout 
  --model_dir checkpoints/qwen3vl_4b \
  --env_name geospot \
  --episodes 1 \
  --batch_size 1 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 5 \
  --max_new_tokens 128 \
  --seed 0
```

**Train**
```bash
# Train on any environment
uv run python core/train.py \
  --model_dir=checkpoints/qwen3vl_4b \
  --env_name=geospot \
  --groups_per_batch=8 \
  --group_size=1 \
  --lr=5e-7 \
  --total_steps=10000
```

**Evaluate**
```bash
uv run python core/eval.py \
  --model_dir checkpoints/qwen3vl_4b \
  --compare_hf \
  --hf_model_name Qwen/Qwen3-VL-4B-Instruct \
  --benchmark_runs 2 \
  --max_new_tokens=128 \
  --prompt "Give me a short introduction to large language models." \
```

Currently the JAX compiler takes a while to run its initial compile, yet the rest of the inference is slightly behind the Hugging Face baseline. TODO: optimize the JAX sampler to improve throughput.

**Preliminary Benchmarks** (Qwen3-VL-4B, A100 80GB, Oct 2024):
| Metric | JAX Sampler | HF Baseline |
| --- | --- | --- |
| Mean tokens/sec | 12.73 ± 0.06 | 16.79 ± 3.03 |
| First-token latency (s) | 28.19 | 0.16 |
| Steady-state throughput (tok/s) | 12.79 | 19.82 |

_First-token latency includes XLA compile time. Reproduce with `core/eval.py --compare_hf --benchmark_runs 2`._

---

## Environments

Creating a custom environment is simple - just extend `envs.base.BaseEnv`:

```python
class MyEnv(BaseEnv):
    def reset(self, idx):
        # Return state and observation
        return state, obs
    
    def step(self, state, action_tokens):
        # Calculate reward based on VLM output
        return state, [], reward, done, info
```

**Built-in environments:**
- `geospot` — **GeoGuessr**: Street-view geolocation with hierarchical rewards (country→region→city→coords)
- `nlvr2` — Two-image True/False reasoning

---

## Requirements

- Python 3.10+
- Linux, CUDA 12, NVIDIA GPU (80GB+ recommended for training; inference requires ~10GB for 4B model)
- JAX 0.6.1 (CUDA 12 build)

---

## References

- **lmpo** — [kvfrans/lmpo](https://github.com/kvfrans/lmpo)
- **Qwen model base** — [jax-ml/jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main/qwen3)
- **NLVR2 dataset** — [HuggingFaceM4/NLVR2](https://huggingface.co/datasets/HuggingFaceM4/NLVR2)

---

## Citation

If you use vlm-gym in your research, please cite:

```bibtex
@software{dantuluri2025vlmgym,
  author = {Dantuluri, Surya},
  title = {vlm-gym: Reinforcement Learning Gym for Vision-Language Models},
  year = {2024},
  url = {https://github.com/sdan/vlm-gym}
}
```

---

## License

See `LICENSE` and `NOTICE`.
