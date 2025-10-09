# vlmrl

is a RL framework for vision–language models written in JAX.

`models/qwen25vl` natively implements Qwen2.5-VL with mRoPE, KV cache, grouped-query attention, and all the nicities you'd want to do inference(`core/sampling.py`), training(`core/grpo.py`), and evaluation(`core/eval.py`).

 make it easy to implement new vision environments(very simple to do in `envs/base.py`) specifically for vision-language models for grounding, computer-use, multimodal reasoning, etc.


## Quickstart
- Install
```bash
uv sync
```
- Convert HF → JAX (defaults to Qwen/Qwen2.5-VL-7B-Instruct)
```bash
python -m utils.hf_to_jax --model_dir checkpoints/qwen25vl_7b
```
Writes `checkpoints/qwen25vl_7b/params.pkl` and tokenizer files.

- Sample
```bash
python -m core.sampling \
  --ckpt_dir checkpoints/qwen25vl_7b \
  --image imgs/f35_takeoff.png \
  --prompt "Describe the image"
```
- Train (GRPO)
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
- Eval
```bash
python core/eval.py \
  --model_dir checkpoints/qwen25vl_7b \
  --env_name=vision \
  --num_generation_tokens=128 \
  --inference_batch_per_device=1 \
  --vlm_max_pixels=1048576 \
  --top_k=5
```

## Environments
- Add your own extending `envs.base.BaseEnv`.
- `vision` / `vision_caption`: single‑image captioning; reward = keyword hits.
- `nlvr2`: two‑image True/False reasoning (downloads via `datasets`).

## Requirements
- Python 3.10+
- Linux, CUDA 12, NVIDIA GPU (≈60GB VRAM for 7B)
- JAX 0.6.1 (CUDA 12 build)
- Qwen2.5-VL 7B-Instruct (for some reason 7b works but 3b doesn't)

---

## References

- **lmpo**: [kvfrans/lmpo](https://github.com/kvfrans/lmpo)
- **Qwen model base**: [jax-ml/jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main/qwen3)
- **NLVR2 dataset**: [HuggingFaceM4/NLVR2](https://huggingface.co/datasets/HuggingFaceM4/NLVR2)

---

## License

See `LICENSE` and `NOTICE` files for details.
