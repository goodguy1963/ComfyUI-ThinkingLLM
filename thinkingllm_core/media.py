"""Media hashing and mask-focus helpers shared by HF and GGUF backends."""

import hashlib

import torch

MASK_FOCUS_INSTRUCTION = (
    "The image has a mask highlight: the selected area is shown at normal brightness and the surrounding "
    "context is dimmed. Analyze and describe only the normal-brightness selected area. Use the dimmed "
    "surroundings only for spatial context and do not describe them."
)

MASK_RECONSTRUCTION_INSTRUCTION = (
    "The selected area is deliberately concealed and contains no usable information about its former "
    "contents. Never identify or describe anything that may have occupied it. Use only the visible "
    "surroundings to infer the most natural scene continuation through that area, and describe the "
    "completed scene without mentioning masks, removal, editing, or missing content."
)


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image with memory optimization"""
    if tensor is None:
        return None
    try:
        if tensor.dim() == 4:
            tensor = tensor[0]

        # More aggressive memory management
        if tensor.is_floating_point():
            # Scale to 0-255 range more efficiently
            tensor = tensor.clamp(0, 1)

        # Reduce sample size for memory efficiency
        array = tensor.cpu().numpy()
        if tensor.numel() > 0:
            # Take smaller sample for hash to save memory
            sample_size = min(50, tensor.numel() // 4)  # Reduced from 100
            sample_pixels = array.flatten()[:sample_size].tolist() if array.size > 0 else []
        else:
            sample_pixels = []

        content = f"{tensor.shape}_{tensor.dtype}_{sample_pixels[:10]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except:
        return None

def get_image_hash(image):
    """Generate hash for image tensor"""
    if image is None:
        return None
    try:
        # Use image tensor properties for hash
        shape = str(image.shape)
        dtype = str(image.dtype)
        # Sample a few pixels for content hash (avoid full tensor for performance)
        if len(image.shape) >= 3:
            sample_pixels = image.flatten()[:100].tolist() if image.numel() > 0 else []
        else:
            sample_pixels = image.flatten().tolist() if image.numel() > 0 else []

        content = f"{shape}_{dtype}_{sample_pixels[:10]}"  # Limit sample size
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except:
        return None


def apply_mask_highlight(image, mask, mode="focus"):
    """Prepare a masked image for focus analysis or background reconstruction."""
    if mask is None:
        return image, None
    if image is None:
        raise ValueError("[QwenVL] A mask requires an image input.")
    if not torch.is_tensor(image) or image.ndim not in (3, 4):
        raise ValueError("[QwenVL] IMAGE must have shape [H,W,C] or [B,H,W,C].")
    if not torch.is_tensor(mask):
        raise ValueError("[QwenVL] MASK must be a tensor.")
    if mode not in {"focus", "reconstruct"}:
        raise ValueError("[QwenVL] mask mode must be 'focus' or 'reconstruct'.")

    if mask.ndim == 2:
        selected_mask = mask
    elif mask.ndim == 3:
        selected_mask = mask[0]
    else:
        raise ValueError("[QwenVL] MASK must have shape [H,W] or [B,H,W].")

    selected_mask = selected_mask.detach().float().clamp(0.0, 1.0)
    if not torch.isfinite(selected_mask).all():
        raise ValueError("[QwenVL] MASK contains non-finite values.")
    if not torch.any(selected_mask > 0):
        raise ValueError("[QwenVL] MASK is empty; select at least one pixel.")

    frame = image[0] if image.ndim == 4 else image
    resized_mask = torch.nn.functional.interpolate(
        selected_mask[None, None],
        size=frame.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].to(device=frame.device, dtype=frame.dtype)
    highlighted_frame = frame.clone()
    color_channels = min(int(frame.shape[-1]), 3)
    if mode == "reconstruct":
        # Fully hide the selected subject, including anti-aliased mask edges, so the
        # vision model can reason from the surrounding scene instead of naming it.
        binary_mask = (resized_mask > 0.01).to(dtype=frame.dtype)
        expanded_mask = torch.nn.functional.max_pool2d(
            binary_mask[None, None], kernel_size=9, stride=1, padding=4,
        )[0, 0]
        visible = 1.0 - expanded_mask
        visible_weight = visible.sum()
        if float(visible_weight.detach().cpu()) > 0:
            fill = (
                frame[..., :color_channels] * visible.unsqueeze(-1)
            ).sum(dim=(0, 1)) / visible_weight
        else:
            fill = torch.full(
                (color_channels,), 0.5, device=frame.device, dtype=frame.dtype,
            )
        highlighted_frame[..., :color_channels] = (
            frame[..., :color_channels] * visible.unsqueeze(-1)
            + fill.view(1, 1, -1) * expanded_mask.unsqueeze(-1)
        )
    else:
        brightness = 0.2 + 0.8 * resized_mask
        highlighted_frame[..., :color_channels] *= brightness.unsqueeze(-1)
    highlighted = image.clone()
    if image.ndim == 4:
        highlighted[0] = highlighted_frame
    else:
        highlighted = highlighted_frame

    digest = hashlib.md5()
    digest.update(mode.encode("ascii"))
    digest.update(str(tuple(mask.shape)).encode("ascii"))
    digest.update(mask.detach().contiguous().cpu().numpy().tobytes())
    return highlighted, digest.hexdigest()[:16]

def get_video_hash(video):
    """Generate hash for video tensor (same as image)"""
    return get_image_hash(video)
