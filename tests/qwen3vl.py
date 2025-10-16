import warnings
import pytest
import jax
import jax.numpy as jnp

# Suppress Pydantic Field warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from transformers import AutoTokenizer

from vlmrl.models.qwen3vl.model import create_model_from_ckpt, VisionEmbeddings
from vlmrl.core.sampling import sample
from vlmrl.core.rollout import collect_episodes, episodes_to_training_batch
from vlmrl.core.train import _resolve_image_pad_id
from vlmrl.core.types import SamplingConfig, VLMInputs
from vlmrl.utils.vlm import (
    preprocess_image,
    chat_prompt_with_image,
    chat_prompt_with_images,
)
from vlmrl.envs.base import create_env


# Baseline specifics (kept simple and consistent):
CKPT_DIR = "checkpoints/qwen3vl_4b"
IMAGE_PATH = "imgs/foxnews.png"
TEMP = 0.7
TOP_P = 0.9
TOP_K = 1024  # baseline; adjust lower locally if needed for speed
SEED = 0


def _build_sampling_config(model, tokenizer, max_new_tokens: int) -> SamplingConfig:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(model.spec, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(model.spec, "pad_token_id", 0) or 0
    return SamplingConfig(
        temperature=float(TEMP),
        top_p=float(TOP_P) if 0.0 < float(TOP_P) < 1.0 else None,
        top_k=int(TOP_K) if TOP_K and int(TOP_K) > 0 else None,
        eos_id=int(eos_id) if eos_id is not None else None,
        pad_id=int(pad_id),
        max_new_tokens=int(max_new_tokens),
    )


def _num_vision_tokens(embeds: VisionEmbeddings | jnp.ndarray) -> int:
    if isinstance(embeds, VisionEmbeddings):
        return int(embeds.tokens.shape[0])
    return int(embeds.shape[0])


def _concat_vision_embeddings(left: VisionEmbeddings | jnp.ndarray, right: VisionEmbeddings | jnp.ndarray):
    if isinstance(left, VisionEmbeddings) and isinstance(right, VisionEmbeddings):
        return VisionEmbeddings.concatenate([left, right])
    return jnp.concatenate([jnp.asarray(left), jnp.asarray(right)], axis=0)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=False)


@pytest.fixture(scope="session")
def model_params():
    model, params = create_model_from_ckpt(CKPT_DIR)
    return model, params

@pytest.fixture(scope="session")
def image_pad_id(tokenizer):
    return _resolve_image_pad_id(tokenizer, CKPT_DIR)


def test_text_sampling_smoke(model_params, tokenizer):
    model, params = model_params
    messages = [{"role": "user", "content": "Say hi in one short sentence."}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    rng = jax.random.PRNGKey(SEED)
    max_new = 8
    cfg = _build_sampling_config(model, tokenizer, max_new)
    result = sample(
        model,
        params,
        prompt_tokens,
        cfg,
        rng,
        tokenizer=tokenizer,
        return_logprobs=True,
    )

    assert result.tokens.shape == (1, max_new)
    assert result.logprobs is not None and result.logprobs.shape == (1, max_new)
    if result.texts:
        assert isinstance(result.texts[0], str)


def test_image_sampling_smoke(model_params, tokenizer, image_pad_id):
    model, params = model_params
    assert model.spec.vision is not None, "Checkpoint must include a vision backbone"

    vision_spec = model.spec.vision
    pixel_values, grid_thw = preprocess_image(
        IMAGE_PATH,
        patch_size=vision_spec.patch_size,
        spatial_merge_size=vision_spec.spatial_merge_size,
        temporal_patch_size=vision_spec.temporal_patch_size,
    )
    vision_embeds = model.apply({"params": params}, pixel_values, grid_thw, method=model.encode_vision)
    num_vis = _num_vision_tokens(vision_embeds)

    prompt = "Describe whats going on in this image"
    prompt_text = chat_prompt_with_image(num_vis, prompt)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    rng = jax.random.PRNGKey(SEED)
    max_new = 8
    cfg = _build_sampling_config(model, tokenizer, max_new)
    inputs = VLMInputs(
        prompt_tokens=prompt_tokens,
        vision=vision_embeds,
        grid_thw=grid_thw,
        image_pad_id=image_pad_id,
    )
    result = sample(
        model,
        params,
        inputs,
        cfg,
        rng,
        tokenizer=tokenizer,
        return_logprobs=True,
    )

    assert result.tokens.shape == (1, max_new)
    assert result.logprobs is not None and result.logprobs.shape == (1, max_new)
    if result.texts:
        assert isinstance(result.texts[0], str)


def test_eval_vlm_smoke(model_params, tokenizer, image_pad_id):
    model, params = model_params
    env = create_env("vision", tokenizer)

    state, prompt = env.reset(0)
    assert prompt.image_path.endswith(".png"), "Vision prompt should point to a demo image"

    vision_spec = model.spec.vision
    assert vision_spec is not None, "Checkpoint must include a vision backbone"
    pixel_values, grid_thw = preprocess_image(
        prompt.image_path,
        patch_size=vision_spec.patch_size,
        spatial_merge_size=vision_spec.spatial_merge_size,
        temporal_patch_size=vision_spec.temporal_patch_size,
    )
    vision_embeds = model.apply({"params": params}, pixel_values, grid_thw, method=model.encode_vision)
    num_vis = _num_vision_tokens(vision_embeds)

    prompt_text = chat_prompt_with_image(num_vis, prompt.question)
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    max_new = 8
    cfg = _build_sampling_config(model, tokenizer, max_new)
    result = sample(
        model,
        params,
        VLMInputs(
            prompt_tokens=prompt_tokens,
            vision=vision_embeds,
            grid_thw=grid_thw,
            image_pad_id=image_pad_id,
        ),
        cfg,
        jax.random.PRNGKey(SEED),
        tokenizer=tokenizer,
        return_logprobs=False,
    )

    assert result.tokens.shape == (1, max_new)
    response_tokens = result.tokens[0].tolist()
    _, _, reward, is_done, infos = env.step(state, response_tokens)
    assert is_done
    assert isinstance(reward, float)
    assert "response" in infos


def test_collect_episodes_basic(model_params, tokenizer, image_pad_id):
    model, params = model_params
    env = create_env("vision", tokenizer)

    cfg = _build_sampling_config(model, tokenizer, max_new_tokens=4)
    rng = jax.random.PRNGKey(SEED)
    batch = collect_episodes(
        env,
        tokenizer,
        model,
        params,
        cfg,
        image_pad_id=image_pad_id,
        batch_size=2,
        rng=rng,
        max_sequence_length=None,
        return_logprobs=True,
    )

    assert len(batch.episodes) == 2
    for episode in batch.episodes:
        assert episode.full_tokens.ndim == 1
        assert isinstance(episode.reward, float)
        assert isinstance(episode.action_tokens, list)

    rollout, train_batch = episodes_to_training_batch(
        model,
        batch.episodes,
        pad_id=cfg.pad_id,
        max_sequence_length=None,
    )

    assert rollout.tokens.shape[0] == 2
    assert rollout.returns.shape[0] == 2
    assert train_batch.tokens.shape == rollout.tokens.shape


def test_ppo_helpers_smoke():
    from vlmrl.core.ppo import _pad_right, _maybe_limit
    import jax.numpy as jnp

    out = _pad_right([jnp.array([[1, 2]]), jnp.array([[3]])], pad_val=0, axis=1)
    assert out.shape[0] == 2
    assert _maybe_limit(-1, 10) is None
    assert _maybe_limit(5, 10) == 5


def test_multi_image_sampling_smoke(model_params, tokenizer, image_pad_id):
    """Covers the two-image path used by NLVR2 and GRPO code paths.

    Uses two local images and chat_prompt_with_images; concatenates embeds and grids.
    """
    model, params = model_params
    assert model.spec.vision is not None
    v = model.spec.vision

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
    vision_embeds = _concat_vision_embeddings(emb_l, emb_r)
    grid_thw = jnp.concatenate([grid_l, grid_r], axis=0)

    num_list = [_num_vision_tokens(emb_l), _num_vision_tokens(emb_r)]
    prompt_text = chat_prompt_with_images(num_list, "True or False?")
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = jnp.asarray([input_ids], dtype=jnp.int32)

    rng = jax.random.PRNGKey(SEED)
    max_new = 8
    cfg = _build_sampling_config(model, tokenizer, max_new)
    result = sample(
        model,
        params,
        VLMInputs(
            prompt_tokens=prompt_tokens,
            vision=vision_embeds,
            grid_thw=grid_thw,
            image_pad_id=image_pad_id,
        ),
        cfg,
        rng,
        tokenizer=tokenizer,
        return_logprobs=True,
    )

    assert result.tokens.shape == (1, max_new)
    assert result.logprobs is not None and result.logprobs.shape == (1, max_new)
