Status: Alpha - expect updates and more documentation. Inference and training run smoothly with >80gb of VRAM GPUs currently with Qwen3-VL-4B-Instruct as the default model.

<img width="325" height="240" alt="vlmgym" src="https://github.com/user-attachments/assets/87d7d141-4464-4687-91c0-3a6da82b2749" />

# vlm-gym

A simple reinforcement learning gym for vision-language models, written in JAX. Drop in any environment, any model, and train with PPO.

**Core components:**
- `envs/` — Pluggable vision environments (GeoGuessr, NLVR2, captioning)
- `models/` — VLM implementations (Qwen3-VL-4B-Instruct reference)
- `core/train.py` — Trainer runs PPO on the environment
- `core/rollout.py` — Inference engine runs the VLM on the environment
- `core/eval.py` — Evaluation harness runs the VLM on the environment and compares it to the Hugging Face baseline

---

## Run a VLM to play GeoGuessr

```bash
uv run python -m core.rollout 
  --model_dir checkpoints/qwen3vl_4b \
  --env_name geospot
```

## Train a VLM to play GeoGuessr

the reward is shaped to improve accuracy on countries, region, cities in that order blended with coordinate depending on the schedule.

```bash
# Train on OpenStreetView-5M dataset
uv run python core/train.py \
  --model_dir checkpoints/qwen3vl_4b \
  --env_name geospot \
  --lr 5e-7 \
  --total_steps 10000
```
---

**Install**
```bash
uv sync
```

**Convert HF → JAX** (defaults to Qwen/Qwen3-VL-4B-Instruct)
```bash
uv run python -m utils.hf_to_jax --model_dir checkpoints/qwen3vl_4b
```

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

Results:
| Metric | JAX Sampler | HF Baseline |
| --- | --- | --- |
| Mean tokens/sec | 12.73 ± 0.06 | 16.79 ± 3.03 |
| First-token latency (s) | 28.19 | 0.16 |
| Steady-state duration (s) | 40.02 | 0.81 |
| Steady-state throughput (tok/s) | 12.79 | 19.82 |

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
- `osv5m` — **GeoGuessr**: Street-view geolocation with hierarchical rewards (country→region→city→coords)
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
