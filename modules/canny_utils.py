import math
from typing import Tuple

import numpy as np


def sanitize_canny_thresholds(low: float, high: float) -> Tuple[float, float]:
    """
    Normalize Canny thresholds to the valid range expected by kornia.

    kornia.filters.canny requires 0 < low, high < 1 and low < high.
    Users can currently set the sliders to 0.0 (or other invalid values),
    which would cause a runtime error. This helper clamps values into a
    safe range and enforces ordering while preserving the intent as much
    as possible.
    """
    eps = 1e-6
    max_val = 1.0 - eps

    # Handle NaNs / infinities by falling back to reasonable defaults.
    if not math.isfinite(low):
        low = eps
    if not math.isfinite(high):
        high = max_val

    # Clamp into the open interval (0, 1).
    if low <= 0.0:
        low = eps
    if high <= 0.0:
        high = max_val

    low = max(eps, min(low, max_val))
    high = max(eps, min(high, max_val))

    # Ensure low < high. When they are equal or inverted, keep the
    # lower bound as-is and nudge the upper bound upwards.
    if not low < high:
        # If low is already near the maximum, move it slightly down
        # so that we still have a valid open interval.
        if low >= max_val:
            low = max_val - eps
            high = max_val
        else:
            high = min(max_val, low + eps)

    return float(low), float(high)


# ---------------------------------------------------------------------------
# PyraCanny
# ---------------------------------------------------------------------------
# Same algorithm Fooocus used (modules/async_worker.py: canny_pyramid), and the
# same one comfyui_controlnet_aux ships as PyraCannyPreprocessor: run Canny at
# scales 0.2 .. 1.0 and accumulate, weighting the running result 0.75 and each
# new scale 0.25. The point is that no single resolution loses structure, which
# is why it holds clothing folds and outlines better than one-scale Canny.
#
# Implemented with cv2 rather than torch/kornia so it needs no new dependency.

PYRA_SCALES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _centered_canny(channel, low, high):
    """Canny on one channel -> float32 in [0, 1]."""
    import cv2

    edge = cv2.Canny(channel, int(low), int(high))
    return edge.astype(np.float32) / 255.0


def _centered_canny_color(rgb, low, high):
    """Canny each channel separately, then stack back to 3 channels."""
    import cv2

    parts = [_centered_canny(rgb[..., i], low, high) for i in range(3)]
    return np.stack(parts, axis=2)


def pyramid_canny_color(rgb, low, high):
    """Accumulate Canny edges across the pyramid, exactly as Fooocus did."""
    import cv2

    height, width, _ = rgb.shape
    acc = None

    for k in PYRA_SCALES:
        small_h, small_w = max(1, int(height * k)), max(1, int(width * k))
        small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
        edge = _centered_canny_color(small, low, high)

        if acc is None:
            acc = edge
        else:
            acc = cv2.resize(
                acc, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            acc = acc * 0.75 + edge * 0.25

    return acc


def _norm255(arr, low=4, high=96):
    """Rescale to 0..255 using percentiles, so a few outliers don't flatten it."""
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo) * 255.0


def canny_pyramid(rgb, low=64, high=128):
    """Fooocus-style PyraCanny.

    rgb: uint8 HxWx3 array.
    low/high: 0..255 thresholds, matching OpenCV.

    Returns a uint8 HxWx3 edge image.
    """
    color_canny = pyramid_canny_color(rgb, low, high)
    result = np.sum(color_canny, axis=2)
    result = _norm255(result, low=1, high=99)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return np.stack([result] * 3, axis=2)


def pyracanny_from_pil(image, low=64, high=128):
    """PIL.Image -> PIL.Image of PyraCanny edges."""
    import numpy as np
    from PIL import Image as _Image

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return _Image.fromarray(canny_pyramid(arr, low, high))


def threshold_from_unit(unit_low, unit_high):
    """Map the UI's 0..1 thresholds onto OpenCV's 0..255 range.

    The UI sliders (and ComfyUI's Canny node) use 0..1, while cv2.Canny wants
    0..255. 0.0 -> 0 and 1.0 -> 255 keeps both ends meaningful.

    Out-of-range or inverted values are clamped so cv2.Canny never gets an
    invalid pair (it silently returns nothing if high <= low).
    """
    def _clamp(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return max(0.0, min(1.0, v))

    lo = _clamp(unit_low)
    hi = _clamp(unit_high)

    if lo is None:
        lo = 0.25
    if hi is None:
        hi = 0.5

    low = int(round(lo * 255))
    high = int(round(hi * 255))

    # Clamp AFTER scaling, so 2.0 and 1.0 both land on 255.
    low = max(0, min(255, low))
    high = max(0, min(255, high))

    if high <= low:
        # Keep a usable window instead of collapsing to a single value.
        if low >= 255:
            low, high = 254, 255
        else:
            high = low + 1

    return low, high



