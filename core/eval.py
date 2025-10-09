import jax.numpy as jnp
import jax
import json
import numpy as np
import tqdm
import ml_collections
import sys
from absl import flags

from transformers import AutoTokenizer

from vlmrl.models.qwen25vl import create_model_from_ckpt
from vlmrl.utils.configs import define_flag_dict
from vlmrl.envs.env_creator import create_env
from vlmrl.utils.sharding import create_sharding, host_gather
from vlmrl.core.sampling import Sampler as VLMSampler, _resolve_image_pad_id
from vlmrl.utils.vlm import (
    preprocess_image,
    chat_prompt_with_image,
    chat_prompt_with_images,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MAX_PIXELS,
)

def eval_model(
    model,
    params,
    env,
    num_generation_tokens,
    force_answer_at,
    prompt_length,
    inference_batch_per_device,
    pad_id,
    shard_data_fn,
    no_shard,
    data_shard,
    num_epochs,
    *,
    tokenizer=None,
    vlm_sampler: VLMSampler | None = None,
    image_pad_id: int | None = None,
    eos_token_id: int | None = None,
    vlm_min_pixels: int | None = None,
    vlm_max_pixels: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float = 1.0,
    max_eval_tasks: int | None = None,
):
    np.random.seed(jax.process_index())
    env_num_tasks = env.num_tasks if env.num_tasks != -1 else 512
    # Optionally cap how many tasks to evaluate for faster, partial runs
    if max_eval_tasks is not None and int(max_eval_tasks) > 0:
        env_num_tasks = min(env_num_tasks, int(max_eval_tasks))
    total_num_tasks = num_epochs * env_num_tasks
    env_task_idx = 0
    rollout_batch_size = jax.local_device_count() * inference_batch_per_device
    global_batch_size = rollout_batch_size * jax.process_count()
    rng = jax.random.PRNGKey(jax.process_index())

    env_infos_history = {}
    env_infos_history['return'] = []
    for i in tqdm.tqdm(range(total_num_tasks // global_batch_size + 1)):
        env_states, env_tokens = [], []
        for _ in range(rollout_batch_size):
            env_state, output_tokens = env.reset(min(env_task_idx + jax.process_index(), env_num_tasks-1))
            env_task_idx += jax.process_count()
            env_task_idx = env_task_idx % env_num_tasks
            env_states.append(env_state)
            env_tokens.append(output_tokens)

        # VLM path only: sample one-by-one via VLMSampler (batch=1 only)
        if vlm_sampler is None or tokenizer is None:
            raise ValueError("VLM sampling requires `vlm_sampler` and `tokenizer`.")

        if image_pad_id is None:
            raise ValueError("VLM sampling requires `image_pad_id`.")

        # Resolve EOS token id if available
        eos_id = eos_token_id
        if eos_id is None and hasattr(tokenizer, "eos_token_id"):
            eos_id = getattr(tokenizer, "eos_token_id")

        action_tokens_list = []
        for obs in env_tokens:
            # Single-image vision caption env
            if hasattr(obs, "image_path") and hasattr(obs, "question"):
                vision_spec = model.spec.vision
                if vision_spec is None:
                    raise ValueError("This VLM checkpoint has no vision backbone configured.")
                pixel_values, grid_thw = preprocess_image(
                    obs.image_path,
                    patch_size=vision_spec.patch_size,
                    spatial_merge_size=vision_spec.spatial_merge_size,
                    temporal_patch_size=vision_spec.temporal_patch_size,
                    min_pixels=(vlm_min_pixels if (vlm_min_pixels and vlm_min_pixels > 0) else DEFAULT_MIN_PIXELS),
                    max_pixels=(vlm_max_pixels if (vlm_max_pixels and vlm_max_pixels > 0) else DEFAULT_MAX_PIXELS),
                )
                vision_embeds = model.apply({"params": params}, pixel_values, grid_thw, method=model.encode_vision)
                num_vision_tokens = int(vision_embeds.shape[0])
                prompt_text = chat_prompt_with_image(num_vision_tokens, obs.question)
                input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

                rng, key = jax.random.split(rng)
                generated, _ = vlm_sampler.sample_vlm(
                    prompt_tokens,
                    vision_embeds,
                    grid_thw,
                    image_pad_id=image_pad_id,
                    max_new_tokens=num_generation_tokens if num_generation_tokens > 0 else env.tokens_per_action,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    rng=key,
                    return_logprobs=False,
                )
                action_tokens_list.append(generated[0].tolist())
            # NLVR2-style two-image prompt
            elif hasattr(obs, "image_left") and hasattr(obs, "image_right") and hasattr(obs, "statement"):
                vision_spec = model.spec.vision
                if vision_spec is None:
                    raise ValueError("This VLM checkpoint has no vision backbone configured.")
                # preprocess both images (PIL or arrays supported)
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
                emb_l = model.apply({"params": params}, pix_l, grid_l, method=model.encode_vision)
                emb_r = model.apply({"params": params}, pix_r, grid_r, method=model.encode_vision)
                vision_embeds = jnp.concatenate([emb_l, emb_r], axis=0)
                grid_thw = jnp.concatenate([grid_l, grid_r], axis=0)
                num_tokens_l = int(emb_l.shape[0])
                num_tokens_r = int(emb_r.shape[0])
                question = f"Look at the two images. {obs.statement} True or False?"
                prompt_text = chat_prompt_with_images([num_tokens_l, num_tokens_r], question)
                input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

                rng, key = jax.random.split(rng)
                generated, _ = vlm_sampler.sample_vlm(
                    prompt_tokens,
                    vision_embeds,
                    grid_thw,
                    image_pad_id=image_pad_id,
                    max_new_tokens=num_generation_tokens if num_generation_tokens > 0 else env.tokens_per_action,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    rng=key,
                    return_logprobs=False,
                )
                action_tokens_list.append(generated[0].tolist())
            else:
                raise ValueError(
                    "This eval only supports VLM envs with `image_path`+`question` or NLVR2 with two images + statement."
                )

        new_states, _, returns_local, dones, env_infos = env.step_list(env_states, action_tokens_list)
        assert dones[0] # Only supports bandit envs for now.
        returns_local = np.array(returns_local)
        returns = host_gather(shard_data_fn(returns_local))
        for k, v in env_infos.items():
            if k not in env_infos_history:
                env_infos_history[k] = []
            np_v = np.array(v)
            # Only shard/gather numeric or boolean arrays; keep strings locally.
            if np_v.dtype.kind in {"i", "u", "f", "b"}:
                v_global = host_gather(shard_data_fn(np_v))
                env_infos_history[k] += v_global.tolist()
            else:
                # Non-numeric (e.g., response text). Extend local list without device collect.
                env_infos_history[k] += list(v)
        env_infos_history['return'] += returns.tolist()
    env_infos_history = {k: np.array(v)[:total_num_tasks] for k, v in env_infos_history.items()}
    return new_states, env_infos_history




######$###########################################
### Runnable function to eval a checkpoint on an env.
##################################################
if __name__ == '__main__':
    config = ml_collections.ConfigDict({
        # env settings.
        'env_name': 'vision',
        'max_eval_tasks': -1,   # when > 0, limit number of tasks evaluated
        'num_generation_tokens': 128, # -1 = use default from env.
        'force_answer_at': -1, # -1 = use default from env.
        'prompt_length': 256, # Length of the prompt to pad to.
        'num_epochs': 1,
        # sampling settings.
        'inference_batch_per_device': 1, # Set this to the maximum until OOM. Should not affect results.
        # sharding mode: 'dp' (replicate) or 'fsdp' (shard params)
        'shard': 'dp',
        # vision preprocessing caps (override to shrink image tokens and avoid OOM)
        'vlm_min_pixels': -1,  # use library default when < 1
        'vlm_max_pixels': 1048576,  # use library default when < 1
        # optional shortlist to reduce softmax work during decoding
        'top_k': -1,           # use full-vocab if < 1
        'top_p': 0.9,          # disabled if <= 0 or >= 1
        'temperature': 0.7,
        # logging & debug controls
        'log_samples': 3,      # print this many response samples
        'log_truncate': 256,   # truncate printed responses to this many chars
        'dump_jsonl': '',      # optional path to write all results as JSONL
        'debug_pdb': 0,        # drop into pdb at end if > 0 (renamed to avoid absl's built-in --pdb)
    })
    define_flag_dict(config)
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
                                            
    ckpt_dir = FLAGS.model_dir
    model, params = create_model_from_ckpt(ckpt_dir)
    rng = jax.random.PRNGKey(0)
    # Use simple DP sharding to reduce peak memory; avoid pre-resharding params
    # Sharding: 'dp' (replicate params) or 'fsdp' (shard params across devices)
    shard_mode = getattr(FLAGS, 'shard', 'dp') if hasattr(FLAGS, 'shard') else 'dp'
    params_shard, no_shard, data_shard, shard_data_fn = create_sharding(shard_mode, params)

    # tokenizer for Qwen2.5-VL
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=False)
    pad_id = getattr(tokenizer, 'pad_token_id', None) or 0
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    # VLM sampler and special token id for image paddings
    vlm_sampler = VLMSampler(model, params)
    image_pad_id = _resolve_image_pad_id(tokenizer, ckpt_dir)

    env = create_env(FLAGS.env_name, tokenizer)
    if FLAGS.num_generation_tokens == -1:
        FLAGS.num_generation_tokens = env.tokens_per_action
    if FLAGS.force_answer_at == -1:
        FLAGS.force_answer_at = env.force_answer_at

    # Derive optional controls
    vlm_min_pixels = int(FLAGS.vlm_min_pixels) if int(FLAGS.vlm_min_pixels) > 0 else None
    vlm_max_pixels = int(FLAGS.vlm_max_pixels) if int(FLAGS.vlm_max_pixels) > 0 else None
    top_k = int(FLAGS.top_k) if int(FLAGS.top_k) > 0 else None
    top_p = float(FLAGS.top_p) if (float(FLAGS.top_p) > 0.0 and float(FLAGS.top_p) < 1.0) else None
    temperature = float(getattr(FLAGS, 'temperature', 1.0) or 1.0)

    new_states, env_infos_history = eval_model(
        model, params, env,
        num_generation_tokens=FLAGS.num_generation_tokens,
        force_answer_at=FLAGS.force_answer_at,
        prompt_length=FLAGS.prompt_length,
        inference_batch_per_device=FLAGS.inference_batch_per_device,
        pad_id=pad_id,
        shard_data_fn=shard_data_fn,
        no_shard=no_shard,
        data_shard=data_shard,
        num_epochs=FLAGS.num_epochs,
        tokenizer=tokenizer,
        vlm_sampler=vlm_sampler,
        image_pad_id=image_pad_id,
        eos_token_id=eos_token_id,
        vlm_min_pixels=vlm_min_pixels,
        vlm_max_pixels=vlm_max_pixels,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_eval_tasks=int(getattr(FLAGS, 'max_eval_tasks', -1) or -1),
    )
    print(" ======================= Example Rollout ======================= ")
    print(new_states[0].render())
    print(" =============================================================== ")
    returns = np.array(env_infos_history.get('return', []), dtype=np.float32)
    print("Number of rollouts:", len(returns))
    if returns.size > 0:
        print(f"return: {float(np.mean(returns))}")
    # Print sample responses in full (truncated)
    responses = env_infos_history.get('response', [])
    if len(responses) > 0:
        n = int(getattr(FLAGS, 'log_samples', 3) or 3)
        trunc = int(getattr(FLAGS, 'log_truncate', 256) or 256)
        print("\nSample responses:")
        for i in range(min(n, len(responses))):
            resp = responses[i]
            text = resp if isinstance(resp, str) else str(resp)
            if len(text) > trunc:
                text = text[:trunc] + "..."
            r = float(returns[i]) if i < len(returns) else float('nan')
            print(f"[{i}] reward={r:.3f} response=\n{text}\n")
    else:
        print("No 'response' field found in env infos.")

    # Optional JSONL dump of all infos
    dump_path = getattr(FLAGS, 'dump_jsonl', '') or ''
    if isinstance(dump_path, str) and len(dump_path) > 0:
        # align by length of the longest field
        keys = list(env_infos_history.keys())
        max_len = max(len(env_infos_history[k]) for k in keys)
        with open(dump_path, 'w') as f:
            for idx in range(max_len):
                rec = {}
                for k in keys:
                    vals = env_infos_history[k]
                    if idx < len(vals):
                        v = vals[idx]
                        # cast numpy types to Python types for JSON
                        if isinstance(v, (np.generic,)):
                            v = v.item()
                        rec[k] = v
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote results to {dump_path}")

    # Optional pdb
    if int(getattr(FLAGS, 'debug_pdb', 0) or 0) > 0:
        breakpoint()
