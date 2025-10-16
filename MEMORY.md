Repo facts (as used by train.py)

  - Batch and sequence knobs: core/train.py:68–82 (batch_size=16, max_new_tokens=64,
    vlm_max_pixels=-1) — vlm_max_pixels=-1 defers to utils/vlm DEFAULT_MAX_PIXELS. core/train.py:68
  - PPO minibatch semantics: “minibatch_size”, not count. If N=16 and ppo_minibatch=64, you still use
    one chunk of size 16. core/ppo.py:187–199 core/ppo.py:187
  - One-hot in PPO loss is present: multiplies [B,T,V] logprobs with a [B,T,V] one_hot. core/
    ppo.py:148–151 core/ppo.py:148
  - Model dtype is bf16, but the vision tower emits fp32; vision embeds are cast to bf16 inside
    forward_vlm. models/qwen3vl/model.py:690–710, models/qwen3vl/vision.py:148–164, models/qwen3vl/
    model.py:793–803 models/qwen3vl/model.py:690 models/qwen3vl/vision.py:148 models/qwen3vl/
    model.py:797
  - Vision padding preserves dtype and includes DeepStack by default (3 extra fp32 arrays). core/
    batching.py:93–111, 113–121 core/batching.py:93
  - Default max pixel budget is very large: DEFAULT_MAX_PIXELS = 12_845_056. utils/vlm.py:17–19
    utils/vlm.py:17

  Verified parameter count (this checkpoint)
  From checkpoints/qwen3vl_4b/config.json:

  - Text: hidden_size=2560, layers=36, heads=32, kv_heads=8, head_dim=128, intermediate=9728,
    vocab=151,936.
  - Vision: hidden_size=1024, out_hidden_size=2560, depth=24, heads=16, intermediate=4096,
    spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[5,11,17].

  Exact param math

  - Attention per text layer (GQA, head_dim=128): q(2560×4096+4096) + k(2560×1024+1024) +
    v(2560×1024+1024) + o(4096×2560) + q_norm(128) + k_norm(128) = 26,220,800
  - MLP per text layer (SiLU): 2560×9728×2 + 9728×2560 = 74,711,040
  - Norms per text layer: 2×2560 = 5,120
  - Text per layer total: 100,936,960 → ×36 layers = 3,633,730,560
  - Embedding + LM head (tied in cfg, but both exist in Flax tree): 2 × (2560×151,936) = 2 ×
    388,956,160 = 777,912,320
  - Final norm: 2,560
  - Text total: 4,411,645,440

  Vision param estimate

  - Patch embed (1,536→1024): 1,572,864
  - 24 blocks: each ≈ 16,783,360 → 402,800,640
  - Main merger: 27,270,656
  - 3× deepstack mergers: 3 × 27,273,728 = 81,821,184
  - Vision total: 513,465,344

  Grand total params: 4,925,110,784

  Parameter memory

  - If float32: 4,925,110,784 × 4 B ≈ 19.70 GB
  - If bf16: 4,925,110,784 × 2 B ≈ 9.85 GB
    Note: your params.pkl is 19.7 GB, which strongly implies float32 storage. There’s no cast-to-bf16
    on load in create_model_from_ckpt, so params sit in fp32 on device.

  Optimizer/grads footprint (AdamW default)

  - AdamW keeps m and v states (~2× params) typically in the same dtype as params. With fp32 params:
      - Params ≈ 19.7 GB
      - Opt state ≈ 39.4 GB
      - Grad tree (ephemeral) ≈ 19.7 GB
        Base training memory from weights + opt + grads ≈ ~78.8 GB before activations/logits/
        workspaces. This alone edges an 80 GB H100.

  Activations (rule of thumb)
  Approx per layer: ~12 × B × T × dmodel × bytes. For dmodel=2560, bf16 activations (2 B):

  - T=600 (single image ~500 vis tokens + ~100 text), B=16:
      - Per layer ≈ 12×16×600×2560×2 B ≈ 0.55 GiB → ×36 ≈ 19.7 GiB
  - T=1000 (two images or very long prompt), B=16:
      - Per layer ≈ 0.92 GiB → ×36 ≈ 33.1 GiB
  - T=2100 (large images at default max pixels), B=16:
      - Per layer ≈ 1.92 GiB → ×36 ≈ 69.1 GiB
        This code does not use gradient checkpointing in the PPO update path, so you’ll pay the full
        backward activation cost.

  Logits + one-hot intermediates in PPO update
  Update computes log_softmax over full vocab and multiplies with one_hot over targets:

  - Logits/logprobs tensor size: B × (T−1) × V with V=151,936; XLA upcasts log_softmax to fp32.
  - One-hot path doubles that footprint while the fusion materializes. From core/ppo.py:148–151 core/
    ppo.py:148
    Examples (fp32):
  - B=16, T≈600: 16×599×151,936×4 B ≈ 5.42 GiB for logprobs; one_hot adds another ≈5.42 GiB; softmax/
    entropy intermediates can add a few more GiB transiently → ~11–16 GiB peak for this step.
  - B=16, T≈1000: ≈9.05 GiB logprobs alone; with one_hot + softmax intermediates → ~18–27 GiB
    transient.
  - B=16, T≈2100: ≈19.0 GiB logprobs; with one_hot + intermediates → ~38–50 GiB transient.

  Vision embeddings and DeepStack buffers

  - Token count per image ≈ (resized area) / (16×16×spatial_merge^2) = (pixels) / 1024
    (temporal_patch_size=2 collapses frames to t=1).
  - With DEFAULT_MAX_PIXELS=12.8M, many OSV5M images won’t be downscaled; e.g., 2048×1024 ≈ 2.1M px →
    ~2,048 vision tokens per image.
  - pad_vision preserves dtype and copies DeepStack (3 extra arrays) by default. core/
    batching.py:103–112 core/batching.py:103
      - Memory ≈ B × tokens × dim × bytes × (1 + deepstack_levels)
      - For B=16, tokens=500, dim=2560, fp32, deepstack=3:
          - ≈ 16×500×2560×4 B × 4 ≈ 313 MiB
      - For tokens=2048 (no downscale):
          - ≈ ~1.25–1.35 GiB

  Why this can blow past 80 GB (in this repo)

  - No reference policy copy in this minimal trainer (you use old_logprobs), so you avoid an extra
    full params copy. However:
  - Params are fp32 today (≈19.7 GB) and AdamW adds ≈39.4 GB for moments; grads add ≈19.7 GB
    transient → ~78.8 GB before activations and logits.
  - Activations during backward (no checkpointing) add ~20–70 GB depending on T.
  - Logits + one_hot + softmax intermediates add ~11–50 GB depending on T and B.
  - Vision embeddings (fp32) + DeepStack staging add ~0.3–1.3 GB.
  - XLA temp buffers (attention workspaces, compilation scratch) need several GB more.

  In short: with batch_size=16, ppo_minibatch=64 (effectively 16), and vlm_max_pixels unset
  (defaulting to very large), this setup will exceed 80 GB even before accounting for XLA scratch
  space.

  Quick knobs to bring it under control

  - Set a sane pixel cap: pass --vlm_max_pixels 120000 (≈117 tokens/image) or 65000 (≈64 tokens).
    This linearly reduces T and both activation and logits memory. See core/train.py:79–81 core/
    train.py:79 and utils/vlm.py:17–19 utils/vlm.py:17.
  - Reduce microbatch size: set --ppo_minibatch 2 or 4. This makes the update path run with B=2 or 4,
    shrinking logits/one_hot memory by 8× or 4×. core/ppo.py:187–206 core/ppo.py:187
  - Kill the one-hot: replace jnp.sum(all_logprobs * jax.nn.one_hot(...), axis=-1) with
    jnp.take_along_axis(all_logprobs, text_target[..., None], -1).squeeze(-1). This avoids
    materializing the [B,T,V] one_hot. core/ppo.py:148–151 core/ppo.py:148
  - Skip DeepStack staging: set VLMRL_SKIP_DEEPSTACK=1 (or use
    core.batching.set_skip_deepstack(True)) to drop 3 fp32 copies in Batch.vision. core/
    batching.py:103–112 core/batching.py:103
  - Prefer Adafactor for optimizer memory: --optimizer adafactor (already wired to bf16 momentum) to
    avoid 2× moments across all params. core/train.py:94–104 core/train.py:94
  - If you can, cast params to bf16 on load to halve param and optimizer state footprint (or keep
    optimizer states in bf16). Your current params.pkl is fp32.

  Rule-of-thumb totals after tweaks (illustrative)

  - With: vlm_max_pixels≈120k (T≈200–300 overall), ppo_minibatch=2, no one_hot, DeepStack skipped,
    Adafactor, bf16 params
      - Params + opt ≈ ~9.85 GB + (~9–10 GB) ≈ ~20 GB
      - Activations (T≈250): ~8–10 GB peak
      - Logits per microbatch (B=2, T≈250): ~0.45–0.9 GB transient
      - Vision embeds (no DeepStack, fp32): ~20–40 MB
      - Headroom for XLA scratch: a few GB
      - Total peak: typically <35 GB, comfortably under 80 GB.