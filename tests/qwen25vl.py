import pytest
import jax
import jax.numpy as jnp

from transformers import AutoTokenizer

from vlmrl.models.qwen25vl import create_model_from_ckpt
from vlmrl.core.sampling import Sampler, _resolve_image_pad_id
from vlmrl.utils.vlm import (
    preprocess_image,
    chat_prompt_with_image,
    chat_prompt_with_images,
)
from vlmrl.core.eval import eval_model
from vlmrl.envs.env_creator import create_env


# Baseline specifics (kept simple and consistent):
CKPT_DIR = "checkpoints/qwen25vl_window"
IMAGE_PATH = "imgs/foxnews.png"
TEMP = 0.7
TOP_P = 0.9
TOP_K = 1024  # baseline; adjust lower locally if needed for speed
SEED = 0


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=False)


@pytest.fixture(scope="session")
def model_params():
    model, params = create_model_from_ckpt(CKPT_DIR)
    return model, params


@pytest.fixture(scope="session")
def sampler(model_params):
    model, params = model_params
    return Sampler(model, params)


@pytest.fixture(scope="session")
def image_pad_id(tokenizer):
    return _resolve_image_pad_id(tokenizer, CKPT_DIR)


def test_text_sampling_smoke(model_params, tokenizer, sampler):
    model, _ = model_params
    eos_id = getattr(tokenizer, "eos_token_id", None) or model.spec.eos_token_id
    pad_id = getattr(tokenizer, "pad_token_id", None) or model.spec.pad_token_id or 0

    messages = [{"role": "user", "content": "Say hi in one short sentence."}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    rng = jax.random.PRNGKey(SEED)
    max_new = 8
    out_tokens, logprobs = sampler.sample_text(
        prompt_tokens,
        max_new_tokens=max_new,
        temperature=TEMP,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_id=eos_id,
        pad_id=pad_id,
        rng=rng,
        return_logprobs=True,
    )

    assert out_tokens.shape == (1, max_new)
    assert logprobs is not None and logprobs.shape == (1, max_new)


def test_image_sampling_smoke(model_params, tokenizer, sampler, image_pad_id):
    model, params = model_params
    assert model.spec.vision is not None, "Checkpoint must include a vision backbone"

    v = model.spec.vision
    pixel_values, grid_thw = preprocess_image(
        IMAGE_PATH,
        patch_size=v.patch_size,
        spatial_merge_size=v.spatial_merge_size,
        temporal_patch_size=v.temporal_patch_size,
    )
    vision_embeds = model.apply({"params": params}, pixel_values, grid_thw, method=model.encode_vision)
    num_vis = int(vision_embeds.shape[0])

    prompt = "Describe whats going on in this image"
    prompt_text = chat_prompt_with_image(num_vis, prompt)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    eos_id = getattr(tokenizer, "eos_token_id", None) or model.spec.eos_token_id
    pad_id = getattr(tokenizer, "pad_token_id", None) or model.spec.pad_token_id or 0

    rng = jax.random.PRNGKey(SEED)
    max_new = 8
    out_tokens, logprobs = sampler.sample_vlm(
        prompt_tokens,
        vision_embeds,
        grid_thw,
        image_pad_id=image_pad_id,
        max_new_tokens=max_new,
        temperature=TEMP,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_id=eos_id,
        pad_id=pad_id,
        rng=rng,
        return_logprobs=True,
    )

    assert out_tokens.shape == (1, max_new)
    assert logprobs is not None and logprobs.shape == (1, max_new)


def test_eval_vlm_smoke(model_params, tokenizer, sampler, image_pad_id):
    model, params = model_params
    env = create_env("vision", tokenizer)

    pad_id = getattr(tokenizer, "pad_token_id", None) or model.spec.pad_token_id or 0
    eos_id = getattr(tokenizer, "eos_token_id", None) or model.spec.eos_token_id

    # identity sharders for a local single-host smoke
    def shard_data_fn(x):
        return x

    _, infos = eval_model(
        model,
        params,
        env,
        num_generation_tokens=8,
        force_answer_at=-1,
        prompt_length=128,
        inference_batch_per_device=1,
        pad_id=pad_id,
        shard_data_fn=shard_data_fn,
        no_shard=None,
        data_shard=None,
        num_epochs=1,
        tokenizer=tokenizer,
        vlm_sampler=sampler,
        image_pad_id=image_pad_id,
        eos_token_id=eos_id,
        vlm_min_pixels=None,
        vlm_max_pixels=None,
        top_k=TOP_K,
        top_p=TOP_P,
        temperature=TEMP,
        max_eval_tasks=1,
    )
    assert "return" in infos and len(infos["return"]) >= 1


def test_grpo_helpers_smoke():
    from vlmrl.core.grpo import _pad_right, _maybe_limit
    import jax.numpy as jnp

    out = _pad_right([jnp.array([[1, 2]]), jnp.array([[3]])], pad_val=0, axis=1)
    assert out.shape[0] == 2
    assert _maybe_limit(-1, 10) is None
    assert _maybe_limit(5, 10) == 5


def test_multi_image_sampling_smoke(model_params, tokenizer, sampler, image_pad_id):
    """Covers the two-image path used by NLVR2 and GRPO code paths.

    Uses two local images and chat_prompt_with_images; concatenates embeds and grids.
    """
    model, params = model_params
    assert model.spec.vision is not None
    v = model.spec.vision

    # Two local images (reuse the same file to keep it simple)
    pix_l, grid_l = preprocess_image(
        IMAGE_PATH,
        patch_size=v.patch_size,
        spatial_merge_size=v.spatial_merge_size,
        temporal_patch_size=v.temporal_patch_size,
    )
    pix_r, grid_r = preprocess_image(
        IMAGE_PATH,
        patch_size=v.patch_size,
        spatial_merge_size=v.spatial_merge_size,
        temporal_patch_size=v.temporal_patch_size,
    )
    emb_l = model.apply({"params": params}, pix_l, grid_l, method=model.encode_vision)
    emb_r = model.apply({"params": params}, pix_r, grid_r, method=model.encode_vision)
    vision_embeds = jnp.concatenate([emb_l, emb_r], axis=0)
    grid_thw = jnp.concatenate([grid_l, grid_r], axis=0)

    prompt_text = chat_prompt_with_images([int(emb_l.shape[0]), int(emb_r.shape[0])], "True or False?")
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    eos_id = getattr(tokenizer, "eos_token_id", None) or model.spec.eos_token_id
    pad_id = getattr(tokenizer, "pad_token_id", None) or model.spec.pad_token_id or 0
    rng = jax.random.PRNGKey(SEED)

    max_new = 6
    out_tokens, logprobs = sampler.sample_vlm(
        prompt_tokens,
        vision_embeds,
        grid_thw,
        image_pad_id=image_pad_id,
        max_new_tokens=max_new,
        temperature=TEMP,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_id=eos_id,
        pad_id=pad_id,
        rng=rng,
        return_logprobs=True,
    )
    assert out_tokens.shape == (1, max_new)
    assert logprobs is not None and logprobs.shape == (1, max_new)


def test_grpo_cli_flags_nlvr2_group5_only_parse():
    """Validate the exact CLI setup parses and passes divisibility checks.

    Does NOT run training; only parses flags and checks rollout divisibility
    for the requested group_size=5 scenario.
    """
    import importlib
    grpo = importlib.import_module("vlmrl.core.grpo")

    # Simulate the CLI you provided, with group_size=5 and batch_per_device=5
    argv = [
        "prog",
        f"--model_dir={CKPT_DIR}",
        "--env_name=nlvr2",
        "--group_size=5",
        "--inference_batch_per_device=5",
        "--do_group_filter=1",
        "--total_steps=1",
        "--test_env_name=",  # keep test env disabled
        "--save_dir=",        # no checkpointing in tests
    ]

    # Parse flags only (no main()); safe in unit tests
    grpo.FLAGS(argv)

    rollout_batch_size = jax.local_device_count() * int(getattr(grpo.FLAGS, "inference_batch_per_device"))
    group_size = int(getattr(grpo.FLAGS, "group_size"))
    assert rollout_batch_size % group_size == 0
