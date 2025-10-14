<img width="225" height="160" alt="vlmgym" src="https://github.com/user-attachments/assets/87d7d141-4464-4687-91c0-3a6da82b2749" />

# vlm-gym

A simple reinforcement learning framework for vision-language models, written in JAX. Drop in any environment, any model, and train with GRPO.

**Core components:**
- `envs/` — Pluggable vision environments (GeoGuessr, NLVR2, captioning)
- `models/` — VLM implementations (Qwen2.5-VL reference)
- `core/grpo.py` — Trainer (currently just a modified version of REINFORCE)
- `core/sampling.py` — Inference engine
- `core/eval.py` — Evaluation harness

---

## train a VLM to play GeoGuessr

the reward is shaped to get countries, then regions, then cities, and finally distance-based correct

```bash
# Train on OpenStreetView-5M dataset
python core/grpo.py \
  --model_dir=checkpoints/qwen25vl_7b \
  --env_name=osv5m \
  --lr=5e-7 \
  --total_steps=10000
```
---

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

**Train**
```bash
# Train on any environment
python core/grpo.py \
  --model_dir=checkpoints/qwen25vl_7b \
  --env_name=osv5m  # or: vision, nlvr2, your_custom_env \
  --groups_per_batch=8 \
  --group_size=1 \
  --lr=5e-7 \
  --total_steps=10000 \
  --wandb_project=vlm-gym
```

**Evaluate**
```bash
python core/eval.py \
  --model_dir checkpoints/qwen25vl_7b \
  --env_name=osv5m  # Match your training env \
  --num_generation_tokens=128 \
  --inference_batch_per_device=1 \
  --vlm_max_pixels=1048576 \
  --top_k=5
```

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
