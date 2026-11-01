"""Realistic photo defaults and a Fooocus-style upscale/enhance stage.

Two things Fooocus does that this fork does not, by default:

  1. A "Realistic" preset that sets photographic styles, CFG and steps.
  2. An "Enhance / Upscale" step after generation, with a Face Swap pass
     folded in (Fooocus's "Enhance" also re-ran face restoration).

Both are optional and off unless asked for, so nothing about the existing
behaviour changes for users who don't touch them.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 1. the Realistic preset
# ---------------------------------------------------------------------------
# Chosen to look like a photograph rather than an illustration:
#   * sai-photographic / sai-cinematic push toward camera optics
#   * lower CFG than the fork's default 8 (8 over-saturates skin at SDXL)
#   * more steps than Speed, fewer than Quality
REALISTIC_PRESET = {
    "style": ["Style: sai-photographic", "Style: sai-cinematic"],
    "performance": "Quality",
    "cfg": 5.0,
    "custom_steps": 40,
    "sampler_name": "dpmpp_2m_sde_gpu",
    "scheduler": "karras",
    "resolution": "896x1152 (3:4)",
    "negative_prompt": (
        "cartoon, anime, illustration, painting, drawing, 3d render, "
        "cgi, plastic skin, airbrushed, oversaturated, deformed, "
        "extra fingers, extra limbs, bad hands, blurry, lowres, "
        "watermark, text, jpeg artifacts"
    ),
}

# ---------------------------------------------------------------------------
# 2. upscale / enhance
# ---------------------------------------------------------------------------
DEFAULT_UPSCALER = "4x-UltraSharp.pth"


def _pil_to_tensor(image):
    """PIL RGB -> torch float tensor (1, C, H, W) in 0..1."""
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_pil(tensor):
    """torch (1, C, H, W) in 0..1 -> PIL RGB."""
    arr = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def load_upscaler(name=DEFAULT_UPSCALER):
    """Load an ESRGAN-family upscaler with spandrel (already a dependency).

    Returns a ModelDescriptor on the GPU, or None if it is unavailable.
    """
    try:
        from spandrel import ModelLoader
    except ImportError:
        print("WARNING: spandrel is not installed; upscaling unavailable")
        return None

    try:
        from shared import path_manager

        path = path_manager.get_file_path(name)
    except Exception as exc:
        print(f"WARNING: could not resolve upscaler {name}: {exc}")
        return None

    if path is None or not Path(path).exists():
        print(f"WARNING: upscaler {name} is not available")
        return None

    try:
        model = ModelLoader().load_from_file(str(path))
        model.eval()
        try:
            model = model.to("cuda")
        except Exception:
            pass          # CPU is slow but must not crash the render
        return model
    except Exception as exc:
        print(f"WARNING: could not load upscaler {name}: {exc}")
        return None


def upscale(image, model=None, name=DEFAULT_UPSCALER):
    """Run an upscale model over a PIL image. Returns PIL, or None on failure."""
    import torch

    if model is None:
        model = load_upscaler(name)
    if model is None:
        return None

    try:
        with torch.no_grad():
            out = model(_pil_to_tensor(image))
        return _tensor_to_pil(out)
    except Exception as exc:
        print(f"WARNING: upscale failed: {exc}")
        return None


def downscale_to(image, target_long_edge):
    """Shrink an upscaled image back to a sane long edge.

    A 4x upscale of a 1024px render is 4096px, which is far more than needed
    for a photo and makes the face swap pass needlessly slow.
    """
    if target_long_edge <= 0:
        return image
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= target_long_edge:
        return image
    scale = target_long_edge / long_edge
    return image.resize(
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        Image.LANCZOS,
    )


def enhance_image(
    image,
    upscale_model_name=DEFAULT_UPSCALER,
    upscale_factor_long_edge=2048,
    swap_target=None,
    swap_kwargs=None,
):
    """One post-process pass: upscale, then optionally re-swap the face.

    Re-swapping after upscaling matters: the 128px swap loses detail when it is
    stretched, so doing it last keeps the identity sharp.

    Returns a PIL image. Any step that fails is skipped, never fatal.
    """
    result = image

    if upscale_model_name and upscale_model_name != "None":
        upscaled = upscale(result, name=upscale_model_name)
        if upscaled is not None:
            result = downscale_to(upscaled, int(upscale_factor_long_edge))

    if swap_target is not None:
        try:
            import modules.faceswap as faceswap

            swapped = faceswap.swap_image(result, swap_target, **(swap_kwargs or {}))
            if swapped is not None:
                result = swapped
        except Exception as exc:
            print(f"WARNING: face swap after upscale failed: {exc}")

    return result
