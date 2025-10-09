"""VLM utilities for sampling, image preprocessing, and tokenization.

"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union, Any

import jax
import jax.numpy as jnp
import numpy as np

# Image preprocessing constants
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
DEFAULT_MIN_PIXELS = 56 * 56
DEFAULT_MAX_PIXELS = 12845056


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    short_edge = min(height, width)
    long_edge = max(height, width)
    if short_edge == 0 or (long_edge / short_edge) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    area = h_bar * w_bar
    if area > max_pixels:
        beta = math.sqrt((height * width) / float(max_pixels))
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif area < min_pixels:
        beta = math.sqrt(float(min_pixels) / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)

    return int(h_bar), int(w_bar)


def preprocess_image(
    image: Union[str, Any],
    *,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate Qwen2.5-VL preprocessing without external HF helpers.

    Accepts either a filesystem path, a PIL.Image.Image, or a numpy array in HWC [0..255] or [0..1].
    """
    import os
    from PIL import Image

    pil_img: Image.Image
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        with Image.open(image) as img:
            pil_img = img.convert("RGB")
    elif hasattr(image, "convert") and hasattr(image, "size"):
        # Likely a PIL Image
        pil_img = image.convert("RGB")
    else:
        # Fallback: assume numpy array HWC
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("Unsupported image array shape; expected HWC with 3 or 4 channels")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        pil_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    width, height = pil_img.size
    factor = patch_size * spatial_merge_size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    if (resized_width, resized_height) != (width, height):
        pil_img = pil_img.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

    image_np = np.asarray(pil_img, dtype=np.float32) / 255.0
    image_np = (image_np - CLIP_MEAN) / CLIP_STD

    # Channels-first, add temporal axis (frames)
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = image_np[None, ...]  # (frames=1, C, H, W)

    if temporal_patch_size > 1:
        frames, _, _, _ = image_np.shape
        remainder = frames % temporal_patch_size
        if remainder != 0:
            pad = temporal_patch_size - remainder
            pad_frame = image_np[-1:, ...]
            image_np = np.concatenate([image_np, np.repeat(pad_frame, pad, axis=0)], axis=0)

    frames, channel, resized_height, resized_width = image_np.shape
    grid_t = frames // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    if grid_h == 0 or grid_w == 0:
        raise ValueError("Resized image too small for given patch size")
    if (grid_h * patch_size != resized_height) or (grid_w * patch_size != resized_width):
        raise ValueError("Resized dimensions must be divisible by patch_size")

    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
        raise ValueError("Grid dimensions must be divisible by spatial_merge_size")

    patches = image_np.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    pixel_values = jnp.asarray(flatten, dtype=jnp.float32)
    grid_thw = jnp.asarray([[grid_t, grid_h, grid_w]], dtype=jnp.int32)
    return pixel_values, grid_thw


def decode_tokens(tokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def chat_prompt_with_image(num_vision_tokens: int, question: str) -> str:
    return (
        f"<|im_start|>user\n"
        f"<|vision_start|>{'<|image_pad|>' * num_vision_tokens}<|vision_end|>"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
    )


def chat_prompt_with_images(num_tokens_list: List[int], question: str) -> str:
    """Build a multi-image chat prompt by concatenating multiple vision blocks.

    Example layout:
    <|im_start|>user
    <|vision_start|>...<|vision_end|>
    <|vision_start|>...<|vision_end|>
    Question text
    <|im_end|>
    <|im_start|>assistant
    """
    vision_blocks = "".join(
        f"<|vision_start|>{'<|image_pad|>' * int(n)}<|vision_end|>" for n in num_tokens_list
    )
    return (
        f"<|im_start|>user\n"
        f"{vision_blocks}"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
    )


def extract_assistant(full_text: str) -> str | None:
    start_marker = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"
    if start_marker not in full_text:
        return None
    start = full_text.rfind(start_marker) + len(start_marker)
    end = full_text.find(end_marker, start)
    if end == -1:
        end = len(full_text)
    return full_text[start:end]


def token_positions(tokens: jnp.ndarray, pad_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # positions for text-only rope
    mask = (tokens != pad_id).astype(jnp.int32)
    positions = jnp.cumsum(mask, axis=-1) - 1
    positions = jnp.where(mask > 0, positions, 0)
    return positions, mask


def apply_top_p_logits(logits: jnp.ndarray, top_p: Optional[float]) -> jnp.ndarray:
    """Exact top-p nucleus filtering (O(V log V)).

    Prefer using `mask_logits_topk_topp` with a reasonable `top_k` to avoid a
    full sort over the vocab each step.
    """
    if top_p is None or not (0.0 < float(top_p) < 1.0):
        return logits
    sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = jax.nn.softmax(sorted_logits.astype(jnp.float32), axis=-1)
    cumprobs = jnp.cumsum(sorted_probs, axis=-1)
    keep_sorted = cumprobs <= jnp.float32(top_p)
    keep_sorted = keep_sorted.at[:, :1].set(True)
    masked_sorted_logits = jnp.where(keep_sorted, sorted_logits, jnp.float32(-1e9))

    def _unsort_row(masked_row_logits, row_indices):
        out = jnp.full_like(masked_row_logits, jnp.float32(-1e9))
        out = out.at[row_indices].set(masked_row_logits)
        return out

    return jax.vmap(_unsort_row)(masked_sorted_logits, sorted_indices)


def mask_logits_topk_topp(
    logits: jnp.ndarray,
    *,
    top_k: int | None,
    top_p: float | None,
) -> jnp.ndarray:
    """Fast masking using top-k shortlist with optional top-p inside the shortlist."""
    vocab = logits.shape[-1]
    use_topk = (top_k is not None) and (int(top_k) > 0)
    use_topp = (top_p is not None) and (0.0 < float(top_p) < 1.0)

    if not use_topk and not use_topp:
        return logits
    if not use_topk and use_topp:
        return apply_top_p_logits(logits, top_p)

    k = int(top_k)
    k = k if k <= vocab else vocab
    top_vals, top_idx = jax.lax.top_k(logits, k)
    if use_topp:
        probs = jax.nn.softmax(top_vals.astype(jnp.float32), axis=-1)
        cumprobs = jnp.cumsum(probs, axis=-1)
        keep = cumprobs <= jnp.float32(top_p)
        keep = keep.at[:, :1].set(True)
        masked_vals = jnp.where(keep, top_vals, jnp.float32(-1e9))
    else:
        masked_vals = top_vals

    out = jnp.full_like(logits, jnp.float32(-1e9))
    batch_idx = jnp.arange(logits.shape[0])[:, None]
    out = out.at[batch_idx, top_idx].set(masked_vals)
    return out


__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "DEFAULT_MIN_PIXELS",
    "DEFAULT_MAX_PIXELS",
    "smart_resize",
    "preprocess_image",
    "decode_tokens",
    "chat_prompt_with_image",
    "chat_prompt_with_images",
    "extract_assistant",
    "token_positions",
    "apply_top_p_logits",
    "mask_logits_topk_topp",
]
