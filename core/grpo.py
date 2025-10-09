import sys
import time
import os
import json
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
import ml_collections
from absl import app, flags

from transformers import AutoTokenizer

from vlmrl.models.qwen25vl import (
    Qwen25VLModel,
    create_model_from_ckpt,
    build_mrope,
    get_rope_index,
)
from vlmrl.utils.configs import define_flag_dict
from vlmrl.utils.wandb import setup_wandb
from vlmrl.envs.env_creator import create_env
from vlmrl.utils.sharding import create_sharding, host_gather
from vlmrl.utils.train_state import TrainState
from vlmrl.utils.checkpoint import Checkpoint
from vlmrl.core.sampling import Sampler as VLMSampler, _resolve_image_pad_id
from vlmrl.utils.vlm import (
    preprocess_image,
    chat_prompt_with_image,
    chat_prompt_with_images,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MAX_PIXELS,
)


# Default config focused on VLM training
config = ml_collections.ConfigDict({
    'wandb_project': "lmpo",
    'wandb_name': 'grpo-vlm',
    'wandb_group': 'Default',
    'model_dir': 'checkpoints/qwen25vl_window',
    'save_dir': "",
    'save_interval': 50,
    # env settings
    'env_name': 'vision',
    'total_steps': 10000,     # number of training steps
    'num_generation_tokens': 64,  # set explicitly for VLM
    # sampling settings
    'inference_batch_per_device': 1,  # keep very small to avoid OOM
    # PPO/GRPO settings
    'groups_per_batch': 8,    # reduce to curb memory
    'group_size': 1,          # ensure rollout_batch_size % group_size == 0 on 1 GPU
    'ppo_minibatch': 2,       # smaller minibatch to reduce logits memory
    'lr': 5e-7,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.001,
    'weight_decay': 1e-2,
    # sharding / optimizer
    'shard': 'dp',            # 'dp' (replicate) or 'fsdp' (shard params)
    'optimizer': 'adafactor', # 'adafactor' (low-memory) or 'adamw'
    # advantage shaping
    'do_group_normalization': 1,
    'do_global_normalization': 0,
    'do_group_filter': 1,
    'do_clip_advantages': 0,
    # importance sampling choices
    'do_inference_logprobs': 0,
    'do_mask_importance_ratio': 0,
    'do_ppo_all_clip': 0,
    # image preprocessing bounds
    'vlm_min_pixels': -1,
    'vlm_max_pixels': -1,
    # decoding controls
    'top_k': -1,
    'top_p': -1.0,
    'temperature': 1.0,
    # optional test pass
    'test_env_name': '',
    'test_interval': 25,
    # debug options
    'debug_print_samples': 0,
    'debug_samples_n': 2,
    'debug_log_wandb_samples': 0,
})
define_flag_dict(config)
FLAGS = flags.FLAGS


def _pad_right(arrs, pad_val, axis=1):
    """Right-pad variable-length sequences to a common length along `axis`.

    Accepts a list of 1D or 2D arrays (or jnp arrays). Returns a stacked jnp array.
    """
    arrays = [jnp.asarray(a) for a in arrs]
    max_len = int(max(a.shape[axis] for a in arrays)) if arrays else 0
    padded = []
    for a in arrays:
        pad_shape = list(a.shape)
        pad_shape[axis] = max_len - a.shape[axis]
        if pad_shape[axis] < 0:
            raise ValueError("_pad_right received an array longer than max_len")
        pad_spec = [(0, 0)] * a.ndim
        pad_spec[axis] = (0, pad_shape[axis])
        padded.append(jnp.pad(a, pad_spec, constant_values=pad_val))
    return jnp.stack(padded, axis=0) if len(padded) > 0 else jnp.zeros((0, max_len), dtype=jnp.int32)


def _maybe_limit(val, default):
    try:
        v = int(val)
        return None if v <= 0 else v
    except Exception:
        return default


def main(_):
    # Init flags and W&B
    FLAGS(sys.argv)
    if jax.process_index() == 0:
        setup_wandb(
            FLAGS.flag_values_dict(),
            project=FLAGS.wandb_project,
            name=f"{FLAGS.env_name}-{FLAGS.wandb_name}",
            group=FLAGS.wandb_group,
        )

    # Load model and optimizer
    ckpt_dir = FLAGS.model_dir
    model, params = create_model_from_ckpt(ckpt_dir)
    # Optimizer: default to Adafactor to dramatically reduce memory vs AdamW.
    if str(getattr(FLAGS, 'optimizer', 'adafactor')).lower() == 'adamw':
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(FLAGS.lr, b1=0.9, b2=0.95, weight_decay=FLAGS.weight_decay),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adafactor(
                learning_rate=FLAGS.lr,
                min_dim_size_to_factor=32,
                dtype_momentum=jnp.bfloat16,
                weight_decay_rate=float(FLAGS.weight_decay) if float(FLAGS.weight_decay) > 0 else None,
            ),
        )
    rng = jax.random.PRNGKey(0)
    init_fn = partial(TrainState.create_with_params, model_def=model, tx=tx, use_ema=False)
    # Create TrainState without JIT to avoid large compile-time IO of params/opt_state.
    train_state = init_fn(rng=rng, params=params)
    # Build sharding for update and helpers. Default to DP to reduce peak memory.
    shard_mode = str(getattr(FLAGS, 'shard', 'dp')).lower()
    train_state_shape = jax.eval_shape(lambda ts: ts, train_state)
    train_state_shard, no_shard, data_shard, shard_data_fn = create_sharding(shard_mode, train_state_shape)

    # Tokenizer and env
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=False)
    pad_id = getattr(tokenizer, 'pad_token_id', None) or 0
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    env = create_env(FLAGS.env_name, tokenizer)
    env_test = create_env(FLAGS.test_env_name, tokenizer) if FLAGS.test_env_name else None

    # Sampler for VLM generation
    vlm_sampler = VLMSampler(model, params)
    image_pad_id = _resolve_image_pad_id(tokenizer, ckpt_dir)

    # Derive min/max pixels and decode shortlist
    vlm_min_pixels = _maybe_limit(FLAGS.vlm_min_pixels, DEFAULT_MIN_PIXELS)
    vlm_max_pixels = _maybe_limit(FLAGS.vlm_max_pixels, DEFAULT_MAX_PIXELS)
    top_k = int(FLAGS.top_k) if int(getattr(FLAGS, 'top_k', -1) or -1) > 0 else None
    top_p = float(FLAGS.top_p)
    top_p = top_p if (0.0 < top_p < 1.0) else None
    temperature = float(getattr(FLAGS, 'temperature', 1.0) or 1.0)

    # GRPO rollout settings
    rollout_batch_size = jax.local_device_count() * int(FLAGS.inference_batch_per_device)
    assert rollout_batch_size % int(FLAGS.group_size) == 0, "rollout_batch_size must be divisible by group_size"
    rng_global = jax.random.PRNGKey(jax.process_index())
    total_rollouts = 0
    env_num_tasks = env.num_tasks if env.num_tasks != -1 else 1_000_000
    env_task_idx = 0

    # JIT compute of token logprobs under current params (teacher-forced)
    def build_get_logprobs_fn(model: Qwen25VLModel):
        @partial(jax.jit, out_shardings=no_shard)
        def get_logprobs(state: TrainState, token_batch, mask, vision_embeds, grid_thw):
            # token_batch: [B, T]
            text_input = token_batch[:, :-1]
            text_target = token_batch[:, 1:]
            mask_target = mask[:, 1:]  # align with targets
            attn_mask = (text_input != pad_id).astype(jnp.int32)
            # Build mRoPE indices for interleaved tokens
            pos3, _ = get_rope_index(
                spatial_merge_size=model.spec.vision.spatial_merge_size,
                input_ids=text_input,
                image_grid_thw=grid_thw,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attn_mask,
                tokens_per_second=float(model.spec.vision.tokens_per_second),
            )
            cos, sin = build_mrope(pos3, tuple(model.spec.text.rope_section), float(model.spec.text.rope_theta), dtype=model.dtype)
            # Forward with vision injection
            logits, _ = state.call_model(text_input, vision_embeds, image_pad_id, cos, sin, mask=attn_mask, cache=None, method=model.forward_vlm)
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            token_logprobs = jnp.sum(logprobs * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)
            # return full sequence logprobs (masked by caller)
            return token_logprobs * mask_target

        return get_logprobs

    get_logprobs = build_get_logprobs_fn(model)

    # Main training loop
    for step_idx in tqdm.tqdm(range(int(FLAGS.total_steps))):
        # On-policy buffer
        buffer_tokens: list[jnp.ndarray] = []
        buffer_adv: list[jnp.ndarray] = []
        buffer_inf_logprobs: list[jnp.ndarray] = []
        buffer_prompt_lens: list[int] = []
        buffer_action_lens: list[int] = []
        buffer_vis_embeds: list[jnp.ndarray] = []
        buffer_grid_thw: list[jnp.ndarray] = []
        buffer_meta: list[dict] = []  # per-group meta for viz snapshot

        env_infos_history = {'return': []}
        rollout_iters = 0
        t_start_roll = time.time()

        while len(buffer_tokens) < int(FLAGS.groups_per_batch):
            rollout_iters += 1
            total_rollouts += rollout_batch_size * jax.process_count()

            # Build a batch of groups (repeat each task group_size times)
            group_env_states = []
            group_obs = []
            for _ in range(rollout_batch_size // int(FLAGS.group_size)):
                env_state, obs = env.reset(min(env_task_idx + jax.process_index(), env_num_tasks - 1))
                env_task_idx = (env_task_idx + jax.process_count()) % env_num_tasks
                for _g in range(int(FLAGS.group_size)):
                    group_env_states.append(env_state)
                    group_obs.append(obs)

            # Per-sample VLM sampling (batch=1 in sampler)
            action_tokens_list: list[list[int]] = []
            all_tokens_list: list[jnp.ndarray] = []
            all_logprobs_list: list[jnp.ndarray] = []
            prompt_lens_list: list[int] = []
            action_lens_list: list[int] = []
            vis_embeds_list: list[jnp.ndarray] = []
            grid_list: list[jnp.ndarray] = []

            for obs in group_obs:
                vision_spec = model.spec.vision
                if vision_spec is None:
                    raise ValueError("This VLM checkpoint has no vision backbone configured.")

                # Build prompt + vision embeddings
                if hasattr(obs, 'image_path') and hasattr(obs, 'question'):
                    pix, grid = preprocess_image(
                        obs.image_path,
                        patch_size=vision_spec.patch_size,
                        spatial_merge_size=vision_spec.spatial_merge_size,
                        temporal_patch_size=vision_spec.temporal_patch_size,
                        min_pixels=(vlm_min_pixels if (vlm_min_pixels and vlm_min_pixels > 0) else DEFAULT_MIN_PIXELS),
                        max_pixels=(vlm_max_pixels if (vlm_max_pixels and vlm_max_pixels > 0) else DEFAULT_MAX_PIXELS),
                    )
                    embeds = model.apply({"params": train_state.params}, pix, grid, method=model.encode_vision)
                    num_tokens = int(embeds.shape[0])
                    prompt_text = chat_prompt_with_image(num_tokens, obs.question)
                    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)
                elif hasattr(obs, 'image_left') and hasattr(obs, 'image_right') and hasattr(obs, 'statement'):
                    pix_l, grid_l = preprocess_image(
                        obs.image_left,
                        patch_size=vision_spec.patch_size,
                        spatial_merge_size=vision_spec.spatial_merge_size,
                        temporal_patch_size=vision_spec.temporal_patch_size,
                        min_pixels=(vlm_min_pixels if (vlm_min_pixels and vlm_min_pixels > 0) else DEFAULT_MIN_PIXELS),
                        max_pixels=(vlm_max_pixels if (vlm_max_pixels and vlm_max_pixels > 0) else DEFAULT_MAX_PIXELS),
                    )
                    pix_r, grid_r = preprocess_image(
                        obs.image_right,
                        patch_size=vision_spec.patch_size,
                        spatial_merge_size=vision_spec.spatial_merge_size,
                        temporal_patch_size=vision_spec.temporal_patch_size,
                        min_pixels=(vlm_min_pixels if (vlm_min_pixels and vlm_min_pixels > 0) else DEFAULT_MIN_PIXELS),
                        max_pixels=(vlm_max_pixels if (vlm_max_pixels and vlm_max_pixels > 0) else DEFAULT_MAX_PIXELS),
                    )
                    emb_l = model.apply({"params": train_state.params}, pix_l, grid_l, method=model.encode_vision)
                    emb_r = model.apply({"params": train_state.params}, pix_r, grid_r, method=model.encode_vision)
                    embeds = jnp.concatenate([emb_l, emb_r], axis=0)
                    grid = jnp.concatenate([grid_l, grid_r], axis=0)
                    prompt_text = chat_prompt_with_images([int(emb_l.shape[0]), int(emb_r.shape[0])], f"Look at the two images. {obs.statement} True or False?")
                    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)
                else:
                    raise ValueError("Unsupported VLM observation structure")

                # Sample
                rng_global, key = jax.random.split(rng_global)
                generated, logprobs = vlm_sampler.sample_vlm(
                    prompt_tokens,
                    embeds,
                    grid,
                    image_pad_id=image_pad_id,
                    max_new_tokens=int(FLAGS.num_generation_tokens),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    rng=key,
                    return_logprobs=True,
                )

                gen_list = generated[0].tolist()
                action_tokens_list.append(gen_list)
                action_lens_list.append(len(gen_list))
                prompt_lens_list.append(int(prompt_tokens.shape[1]))
                vis_embeds_list.append(embeds)
                grid_list.append(grid)

                all_tokens = jnp.concatenate([prompt_tokens, generated], axis=-1)
                all_tokens_list.append(all_tokens[0])
                # prepend zeros for prompt region
                zeros = jnp.zeros((prompt_tokens.shape[1],), dtype=jnp.float32)
                all_logprobs = jnp.concatenate([zeros, logprobs[0]], axis=0)
                all_logprobs_list.append(all_logprobs)

            # Step environment
            new_states, _, returns_local, dones, env_infos = env.step_list(group_env_states, action_tokens_list)
            assert dones[0]
            returns_local = np.array(returns_local)
            returns = host_gather(shard_data_fn(returns_local))

            for k, v in env_infos.items():
                env_infos_history.setdefault(k, [])
                np_v = np.array(v)
                if np_v.dtype.kind in {'i', 'u', 'f', 'b'}:
                    v_global = host_gather(shard_data_fn(np_v))
                    env_infos_history[k] += v_global.tolist()
                else:
                    env_infos_history[k] += list(v)
            env_infos_history['return'] += returns.tolist()

            # Optional debug print/log of a few samples
            if int(getattr(FLAGS, 'debug_print_samples', 0)):
                try:
                    n_dbg = int(getattr(FLAGS, 'debug_samples_n', 2) or 2)
                    n_dbg = min(n_dbg, len(group_obs))
                    responses = env_infos.get('response', [''] * len(group_obs))
                    for i in range(n_dbg):
                        o = group_obs[i]
                        resp = responses[i] if i < len(responses) else ''
                        kws = getattr(o, 'keywords', ())
                        q = getattr(o, 'question', '')
                        img = getattr(o, 'image_path', '')
                        rew = float(returns_local[i]) if i < len(returns_local) else float('nan')
                        print("\n[DEBUG SAMPLE]")
                        print(f"Q: {q}")
                        if img:
                            print(f"Image: {img}")
                        print(f"Keywords: {', '.join(kws) if kws else '(none)'}")
                        print(f"Response: {resp}")
                        print(f"Reward: {rew:.3f}")
                except Exception as _e:
                    # Keep debug non-fatal
                    pass

            if int(getattr(FLAGS, 'debug_log_wandb_samples', 0)) and jax.process_index() == 0:
                try:
                    import wandb as _wandb
                    n_dbg = int(getattr(FLAGS, 'debug_samples_n', 2) or 2)
                    n_dbg = min(n_dbg, len(group_obs))
                    responses = env_infos.get('response', [''] * len(group_obs))
                    rows = []
                    for i in range(n_dbg):
                        o = group_obs[i]
                        resp = responses[i] if i < len(responses) else ''
                        q = getattr(o, 'question', '')
                        kws = ', '.join(getattr(o, 'keywords', ()))
                        img = getattr(o, 'image_path', '')
                        rew = float(returns_local[i]) if i < len(returns_local) else float('nan')
                        rows.append([int(step_idx), q, kws, resp, rew, img])
                    if rows:
                        tbl = _wandb.Table(columns=[
                            'step', 'question', 'keywords', 'response', 'reward', 'image_path'
                        ], data=rows)
                        _wandb.log({'debug/samples': tbl}, commit=False)
                except Exception:
                    pass

            # Compute advantages per group
            returns = jnp.reshape(returns, (-1, int(FLAGS.group_size)))
            adv = returns
            if int(FLAGS.do_group_normalization):
                gm = np.mean(adv, axis=-1)
                gs = np.std(adv, axis=-1) + 1e-8
                adv = (adv - gm[:, None]) / gs[:, None]
            if int(FLAGS.do_global_normalization):
                gmean = np.mean(adv)
                gstd = np.std(adv) + 1e-8
                adv = (adv - gmean) / gstd
            if int(FLAGS.do_clip_advantages):
                adv = np.clip(adv, a_min=0.0, a_max=None)

            # Flatten back and filter groups where all adv == 0 (optional)
            all_tokens_arr = _pad_right(all_tokens_list, pad_val=pad_id, axis=0)
            all_logprobs_arr = _pad_right(all_logprobs_list, pad_val=0.0, axis=0)
            adv_grouped = adv.reshape(-1, int(FLAGS.group_size))
            tokens_grouped = all_tokens_arr.reshape(-1, int(FLAGS.group_size), all_tokens_arr.shape[-1])
            logprobs_grouped = all_logprobs_arr.reshape(-1, int(FLAGS.group_size), all_logprobs_arr.shape[-1])
            prompt_lens_grouped = np.array(prompt_lens_list).reshape(-1, int(FLAGS.group_size))
            action_lens_grouped = np.array(action_lens_list).reshape(-1, int(FLAGS.group_size))
            vis_embeds_grouped = np.array(vis_embeds_list, dtype=object).reshape(-1, int(FLAGS.group_size))
            grid_grouped = np.array(grid_list, dtype=object).reshape(-1, int(FLAGS.group_size))
            returns_grouped = np.array(returns).reshape(-1, int(FLAGS.group_size))

            # Capture per-sample env outputs for this rollout
            responses = env_infos.get('response', [''] * len(group_obs)) if isinstance(env_infos, dict) else [''] * len(group_obs)
            predicted_flags = env_infos.get('predicted', [None] * len(group_obs)) if isinstance(env_infos, dict) else [None] * len(group_obs)

            for g in range(adv_grouped.shape[0]):
                if int(FLAGS.do_group_filter) and np.all(adv_grouped[g] == 0):
                    continue
                buffer_tokens.append(tokens_grouped[g])
                buffer_adv.append(adv_grouped[g])
                buffer_inf_logprobs.append(logprobs_grouped[g])
                buffer_prompt_lens.append(int(prompt_lens_grouped[g, 0]))  # same per group
                buffer_action_lens.append(int(action_lens_grouped[g, 0]))  # same per group
                # store one set per group (identical across members)
                buffer_vis_embeds.append(vis_embeds_grouped[g, 0])
                buffer_grid_thw.append(grid_grouped[g, 0])
                # snapshot meta (use the first obs of this group)
                try:
                    base_idx = g * int(FLAGS.group_size)
                    o = group_obs[base_idx]
                    meta = {
                        'statement': getattr(o, 'statement', ''),
                        'label': bool(getattr(o, 'label', False)),
                        'image_left': getattr(o, 'image_left', None),
                        'image_right': getattr(o, 'image_right', None),
                        'returns': returns_grouped[g].tolist(),
                        'advantages': adv_grouped[g].tolist(),
                        'responses': [responses[base_idx + k] if base_idx + k < len(responses) else '' for k in range(int(FLAGS.group_size))],
                        'predicted': [bool(predicted_flags[base_idx + k]) if base_idx + k < len(predicted_flags) and predicted_flags[base_idx + k] is not None else None for k in range(int(FLAGS.group_size))],
                    }
                    buffer_meta.append(meta)
                except Exception:
                    buffer_meta.append({'statement': '', 'label': None, 'image_left': None, 'image_right': None, 'returns': [], 'advantages': [], 'responses': [], 'predicted': []})

            if jax.process_index() == 0:
                print(f"Buffer groups: {len(buffer_tokens)}. Return avg: {float(np.mean(returns)):.4f}")

        # Construct global batch tensors (pad to right)
        tokens_all = jnp.concatenate(buffer_tokens, axis=0)
        inference_logprobs_all = _pad_right(buffer_inf_logprobs, pad_val=0.0, axis=1)
        advantages_all = jnp.concatenate(buffer_adv, axis=0)
        global_batch_size = int(FLAGS.groups_per_batch) * int(FLAGS.group_size)
        # Clip to exact global batch
        tokens_all = tokens_all[:global_batch_size]
        advantages_all = advantages_all[:global_batch_size]
        inference_logprobs_all = inference_logprobs_all[:global_batch_size]

        # Build generation mask per-sample (only learned on generated tokens)
        prompt_lens_arr = np.array(buffer_prompt_lens, dtype=np.int32)
        action_lens_arr = np.array(buffer_action_lens, dtype=np.int32)
        prompt_lens_arr = np.repeat(prompt_lens_arr, int(FLAGS.group_size))[:global_batch_size]
        action_lens_arr = np.repeat(action_lens_arr, int(FLAGS.group_size))[:global_batch_size]
        T = int(tokens_all.shape[-1])
        mask = np.zeros((global_batch_size, T), dtype=np.int32)
        for b in range(global_batch_size):
            st = int(prompt_lens_arr[b]) - 1
            ln = max(int(action_lens_arr[b]) - 1, 0)
            ed = min(st + ln, T - 1)
            if st >= 0 and ed >= st:
                mask[b, st:ed + 1] = 1
        mask = jnp.asarray(mask, dtype=jnp.int32)

        # Minibatch over flat indices; build per-sample vision context
        mb_size = int(FLAGS.ppo_minibatch)
        mb_count = (global_batch_size + mb_size - 1) // mb_size

        def pack_vision_samples(sample_indices):
            # Map sample index -> group index, then replicate group vision embeds per sample
            embeds_list = []
            grid_list_local = []
            for sidx in sample_indices:
                gidx = int(sidx) // int(FLAGS.group_size)
                embeds_list.append(buffer_vis_embeds[gidx])
                grid_list_local.append(buffer_grid_thw[gidx])
            emb_batch = _pad_right(embeds_list, pad_val=0.0, axis=0)
            grd_batch = _pad_right(grid_list_local, pad_val=0, axis=0)
            return emb_batch, grd_batch

        # First pass: recompute logprobs per minibatch
        recalc_chunks = []
        for j in range(mb_count):
            st = j * mb_size
            ed = min((j + 1) * mb_size, global_batch_size)
            idxs = list(range(st, ed))
            tokens_mb = tokens_all[st:ed]
            mask_mb = mask[st:ed]
            embeds_mb, grid_mb = pack_vision_samples(idxs)
            recalc = get_logprobs(train_state, tokens_mb, mask_mb, embeds_mb, grid_mb)
            recalc_chunks.append(recalc)
        recalc_logprobs_all = jnp.concatenate(recalc_chunks, axis=0)

        # Optional: emit a viz snapshot JSON for the first buffered group
        snapshot_out = os.environ.get('GRPO_SNAPSHOT_OUT')
        if snapshot_out and buffer_tokens:
            try:
                group_index = int(os.environ.get('GRPO_SNAPSHOT_GROUP', '0') or 0)
            except Exception:
                group_index = 0
            group_index = max(0, min(group_index, len(buffer_tokens) - 1))
            # Build per-sample indices for this group in the flattened arrays
            gsz = int(FLAGS.group_size)
            s_start = group_index * gsz
            s_end = s_start + gsz
            # compute per-sample ratios (exp of sum of logprob deltas over mask)
            inf_mb = _pad_right([buffer_inf_logprobs[group_index]], pad_val=0.0, axis=1)[0]
            rec_mb = recalc_logprobs_all[s_start:s_end]
            mask_mb = mask[s_start:s_end]
            ratio_list = []
            for b in range(gsz):
                dlp = (rec_mb[b] - inf_mb[b]) * mask_mb[b]
                ratio = float(np.exp(np.sum(np.asarray(dlp))))
                ratio_list.append(ratio)
            eps = float(getattr(FLAGS, 'clip_epsilon', 0.2) or 0.2)
            # Save images (if PIL) next to snapshot
            out_path = Path(snapshot_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            meta = buffer_meta[group_index] if group_index < len(buffer_meta) else {}
            stmt = meta.get('statement', '')
            img_l = meta.get('image_left', None)
            img_r = meta.get('image_right', None)
            img_l_path = None
            img_r_path = None
            try:
                if img_l is not None and hasattr(img_l, 'save'):
                    img_l_path = out_path.parent / f"group{group_index}_left.png"
                    img_l.save(img_l_path)
                    img_l_path = str(img_l_path)
            except Exception:
                img_l_path = None
            try:
                if img_r is not None and hasattr(img_r, 'save'):
                    img_r_path = out_path.parent / f"group{group_index}_right.png"
                    img_r.save(img_r_path)
                    img_r_path = str(img_r_path)
            except Exception:
                img_r_path = None
            # Assemble samples
            rewards = meta.get('returns', [])
            advantages = meta.get('advantages', [])
            responses = meta.get('responses', [])
            predicted = meta.get('predicted', [])
            samples = []
            for i in range(gsz):
                r = float(rewards[i]) if i < len(rewards) else 0.0
                a = float(advantages[i]) if i < len(advantages) else 0.0
                ratio = ratio_list[i] if i < len(ratio_list) else 1.0
                ratio_clip = float(np.clip(ratio, 1.0 - eps, 1.0 + eps))
                within_clip = bool(abs(1.0 - ratio) <= eps)
                samples.append({
                    'sample_idx': i,
                    'image_files': [img_l_path, img_r_path],
                    'reward': r,
                    'advantage': a,
                    'ppo_ratio': ratio,
                    'ratio_clipped': ratio_clip,
                    'within_clip': within_clip,
                    'prediction_text': responses[i] if i < len(responses) else None,
                    'predicted': (bool(predicted[i]) if i < len(predicted) and predicted[i] is not None else None),
                })
            # Baseline = true group mean reward
            baseline = float(np.mean(np.asarray(rewards))) if rewards else 0.0
            adv_flavor = 'group_norm' if int(FLAGS.do_group_normalization) else ('global_norm' if int(FLAGS.do_global_normalization) else 'raw')
            snapshot = {
                'step': int(step_idx),
                'env': str(FLAGS.env_name),
                'group_id': int(group_index),
                'statement': stmt,
                'baseline_mean_reward': baseline,
                'advantage_flavor': adv_flavor,
                'samples': samples,
                'log_lines': [],
                'label': meta.get('label', None),
            }
        else:
            snapshot = None

        # JIT PPO update using text-style objective but VLM logprobs
        @partial(jax.jit, out_shardings=(train_state_shard, None))
        def update(state: TrainState, token_batch, mask_origin, advantages, recalc_logprobs, inference_logprobs, vision_embeds_mb, grid_thw_mb):
            text_input, text_target = token_batch[:, :-1], token_batch[:, 1:]
            inference_logprobs = inference_logprobs[:, 1:]
            mask_origin = mask_origin[:, 1:]
            token_mask = (text_input != pad_id).astype(jnp.int32)

            def loss_fn(grad_params):
                # Recompute logits (teacher-forced)
                pos3, _ = get_rope_index(
                    spatial_merge_size=model.spec.vision.spatial_merge_size,
                    input_ids=text_input,
                    image_grid_thw=grid_thw_mb,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=token_mask,
                    tokens_per_second=float(model.spec.vision.tokens_per_second),
                )
                cos, sin = build_mrope(pos3, tuple(model.spec.text.rope_section), float(model.spec.text.rope_theta), dtype=model.dtype)
                logits, _ = state.call_model(text_input, vision_embeds_mb, image_pad_id, cos, sin, mask=token_mask, cache=None, params=grad_params, method=model.forward_vlm)
                all_logprobs = jax.nn.log_softmax(logits)
                token_logprobs = jnp.sum(all_logprobs * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)
                entropy = -jnp.sum(jax.nn.softmax(logits) * all_logprobs, axis=-1)

                old_logprobs = jnp.where(int(FLAGS.do_inference_logprobs) == 1, inference_logprobs, recalc_logprobs)
                logratio = token_logprobs - old_logprobs
                ratio = jnp.exp(logratio)
                if int(FLAGS.do_ppo_all_clip):
                    pg_loss = -advantages[:, None] * jnp.clip(ratio, 1 - FLAGS.clip_epsilon, 1 + FLAGS.clip_epsilon)
                else:
                    pg_loss1 = -advantages[:, None] * ratio
                    pg_loss2 = -advantages[:, None] * jnp.clip(ratio, 1 - FLAGS.clip_epsilon, 1 + FLAGS.clip_epsilon)
                    pg_loss = jnp.maximum(pg_loss1, pg_loss2)

                def avg_over_mask(x):
                    return jnp.sum(x * mask_origin) / (jnp.sum(mask_origin) + 1e-8)
                entropy_avg = avg_over_mask(entropy)
                loss_pg = jnp.mean(pg_loss * mask_origin)
                loss_ent = -entropy_avg * float(FLAGS.entropy_coef)
                loss = loss_pg + loss_ent
                return loss, {
                    'loss': loss,
                    'loss_pg': loss_pg,
                    'loss_ent': loss_ent,
                    'entropy': entropy_avg,
                }

            grads, info = jax.grad(loss_fn, has_aux=True)(state.params)
            updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            state = state.replace(params=new_params, opt_state=opt_state, step=state.step + 1)
            info['grad_norm'] = optax.global_norm(grads)
            info['update_norm'] = optax.global_norm(updates)
            info['param_norm'] = optax.global_norm(new_params)
            return state, info

        # Training loop over minibatches
        t_update_start = time.time()
        for j in range(mb_count):
            st = j * mb_size
            ed = min((j + 1) * mb_size, global_batch_size)
            idxs = list(range(st, ed))
            tokens_mb = tokens_all[st:ed]
            mask_mb = mask[st:ed]
            adv_mb = advantages_all[st:ed]
            inf_lp_mb = inference_logprobs_all[st:ed]
            recalc_mb = recalc_logprobs_all[st:ed]
            embeds_mb, grid_mb = pack_vision_samples(idxs)
            train_state, info = update(
                train_state,
                tokens_mb,
                mask_mb,
                adv_mb,
                recalc_mb,
                inf_lp_mb,
                embeds_mb,
                grid_mb,
            )
            info = jax.device_get(info)
            info = jax.tree.map(lambda x: np.array(x), info)
            info = jax.tree.map(lambda x: x.mean(), info)
            info['total_rollouts'] = total_rollouts
            if env.num_tasks != -1:
                info['env_epochs'] = total_rollouts / env_num_tasks
            info['rollout_iters_per_update'] = rollout_iters
            info['global_step'] = step_idx
            info['times/time_per_inference_iteration'] = (time.time() - t_start_roll) / max(rollout_iters, 1)
            info['times/time_per_rollout'] = (time.time() - t_start_roll) / max(rollout_iters * rollout_batch_size * jax.host_count(), 1)
            info['times/total_time_rollouts'] = time.time() - t_start_roll
            info['times/total_time_update'] = time.time() - t_update_start
            info['minibatches_per_global_step'] = int(mb_count)
            for k, v in env_infos_history.items():
                if isinstance(v, (list, tuple)):
                    vals = v
                else:
                    vals = list(v)
                # avoid large text blobs in logs; only mean for numeric
                try:
                    vals_np = np.array(vals)
                    if vals_np.dtype.kind in {'i', 'u', 'f'}:
                        info[f'env/{k}'] = float(np.mean(vals_np))
                except Exception:
                    pass
            if jax.process_index() == 0:
                wandb.log(info)
                # If building a viz snapshot, append a few numeric lines now that we have update stats
                if snapshot is not None and snapshot.get('log_lines') is not None and len(snapshot['log_lines']) < 1:
                    try:
                        gn = float(info.get('grad_norm', 0.0))
                        un = float(info.get('update_norm', 0.0))
                        snapshot['grad_norm'] = gn
                        snapshot['update_norm'] = un
                        snapshot['log_lines'] += [
                            f"baseline_mean_reward={snapshot.get('baseline_mean_reward', 0.0):.3f}",
                            f"grad_norm={gn:.3f}",
                            f"update_norm={un:.3f}",
                        ]
                    except Exception:
                        pass

        # Finally, write snapshot JSON once per step (if requested)
        if snapshot is not None and jax.process_index() == 0:
            try:
                with open(os.environ['GRPO_SNAPSHOT_OUT'], 'w') as f:
                    json.dump(snapshot, f)
                print(f"[viz] wrote GRPO snapshot → {os.environ['GRPO_SNAPSHOT_OUT']}")
            except Exception as e:
                print(f"[viz] failed to write GRPO snapshot: {e}")

        # Optional eval on a test env
        if step_idx % int(FLAGS.test_interval) == 0 and env_test is not None:
            # Run a quick single batch of rollouts and log avg reward
            states, obs = [], []
            for _ in range(rollout_batch_size):
                s, o = env_test.reset(min(env_task_idx + jax.process_index(), env_num_tasks - 1))
                env_task_idx = (env_task_idx + jax.process_count()) % env_num_tasks
                states.append(s)
                obs.append(o)
            actions = []
            for o in obs:
                vision_spec = model.spec.vision
                if hasattr(o, 'image_path') and hasattr(o, 'question'):
                    pix, grid = preprocess_image(
                        o.image_path,
                        patch_size=vision_spec.patch_size,
                        spatial_merge_size=vision_spec.spatial_merge_size,
                        temporal_patch_size=vision_spec.temporal_patch_size,
                    )
                    embeds = model.apply({"params": train_state.params}, pix, grid, method=model.encode_vision)
                    num_tokens = int(embeds.shape[0])
                    prompt_text = chat_prompt_with_image(num_tokens, o.question)
                    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)
                    rng_local = jax.random.PRNGKey(step_idx)
                    gen, _ = vlm_sampler.sample_vlm(
                        prompt_tokens, embeds, grid, image_pad_id,
                        max_new_tokens=int(FLAGS.num_generation_tokens),
                        temperature=temperature, top_p=top_p, top_k=top_k,
                        eos_id=eos_id, pad_id=pad_id, rng=rng_local, return_logprobs=False,
                    )
                    actions.append(gen[0].tolist())
                else:
                    # Minimal fallback: no support for multi-image in quick test here
                    actions.append([])
            _, _, ret_local, _, _ = env_test.step_list(states, actions)
            ret_local = np.array(ret_local)
            ret = host_gather(shard_data_fn(ret_local))
            if jax.process_index() == 0:
                wandb.log({'test_env/return': float(np.mean(ret))}, commit=False)

        # Optional checkpoint save (params only)
        if step_idx % int(FLAGS.save_interval) == 0 and FLAGS.save_dir:
            params_gather = host_gather(train_state.params)
            if jax.process_index() == 0:
                step_dir = FLAGS.save_dir + f"/step{step_idx}/"
                cp = Checkpoint(step_dir + 'params.pkl', parallel=False)
                cp.params = params_gather
                cp.save()


if __name__ == '__main__':
    app.run(main)
