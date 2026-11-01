"""Verify PyraCanny matches the reference implementations.

Compares modules/canny_utils.py against:
  1. comfyui_controlnet_aux's PyraCannyPreprocessor (the canonical one)
  2. Fooocus's canny_pyramid (same algorithm)

They must agree exactly, not approximately: this is a port, not a reimplementation.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"E:\go\face swap\RuinedFooocus")
sys.path.insert(0, str(REPO))

import cv2
from modules import canny_utils as cu

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


# ---------------------------------------------------------------------------
# Reference implementation, copied verbatim from
# comfyui_controlnet_aux/src/custom_controlnet_aux/pyracanny/__init__.py
# ---------------------------------------------------------------------------
def ref_centered_canny(x, low, high):
    y = cv2.Canny(x, int(low), int(high))
    return y.astype(np.float32) / 255.0


def ref_centered_canny_color(x, low, high):
    result = [ref_centered_canny(x[..., i], low, high) for i in range(3)]
    return np.stack(result, axis=2)


def ref_pyramid_canny_color(x, low, high):
    H, W, C = x.shape
    acc_edge = None
    for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        Hs, Ws = int(H * k), int(W * k)
        small = cv2.resize(x, (Ws, Hs), interpolation=cv2.INTER_AREA)
        edge = ref_centered_canny_color(small, low, high)
        if acc_edge is None:
            acc_edge = edge
        else:
            acc_edge = cv2.resize(
                acc_edge, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            acc_edge = acc_edge * 0.75 + edge * 0.25
    return acc_edge


def ref_norm255(x, low=4, high=96):
    v_min = np.percentile(x, low)
    v_max = np.percentile(x, high)
    x -= v_min
    x /= v_max - v_min
    return x * 255.0


def ref_canny_pyramid(x, low, high):
    color_canny = ref_pyramid_canny_color(x, low, high)
    result = np.sum(color_canny, axis=2)
    return ref_norm255(result, low=1, high=99).clip(0, 255).astype(np.uint8)


print("=" * 66)
print("1) byte-identical to the reference implementation")
print("=" * 66)

rng = np.random.default_rng(0)
cases = []
# a synthetic photo-ish image with structure
base = np.zeros((256, 192, 3), np.uint8)
for y in range(0, 256, 32):
    base[y:y + 16, :, 0] = 200
for x in range(0, 192, 24):
    base[:, x:x + 12, 1] = 140
base[64:192, 48:144, 2] = 90
cases.append(("synthetic blocks", base))
cases.append(("random noise", rng.integers(0, 256, (200, 150, 3), dtype=np.uint8)))
cases.append(("gradient", np.dstack([
    np.tile(np.linspace(0, 255, 180, dtype=np.uint8), (240, 1)),
    np.tile(np.linspace(0, 255, 180, dtype=np.uint8)[::-1], (240, 1)),
    np.full((240, 180), 128, np.uint8),
])))

for name, img in cases:
    for low, high in [(64, 128), (50, 100), (100, 200)]:
        mine = cu.canny_pyramid(img, low, high)
        theirs = ref_canny_pyramid(img.copy(), low, high)
        # The reference returns a 2-D edge map; ours repeats it across 3
        # channels because ComfyUI's IMAGE type is HxWx3. Compare the map.
        mine2d = mine[..., 0]
        identical = np.array_equal(mine2d, theirs)
        if not identical:
            diff = np.abs(mine2d.astype(int) - theirs.astype(int))
            print(f"    {name} ({low},{high}): max diff {diff.max()}, "
                  f"{np.count_nonzero(diff)} px differ")
        check(identical, f"{name} low={low} high={high} is identical")

print()
print("=" * 66)
print("2) shape / dtype / range contracts")
print("=" * 66)
img = cases[0][1]
out = cu.canny_pyramid(img, 64, 128)
check(out.shape == img.shape, f"same HxWx3 shape ({out.shape})")
check(out.dtype == np.uint8, f"uint8 output ({out.dtype})")
check(out.max() > 0, f"produced actual edges (max {out.max()})")
check(len(np.unique(out[..., 0])) > 5, "more than a couple of grey levels (real gradient)")

# all three channels equal, as the reference sums them into one map
check(np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2]),
      "all three channels carry the same edge map")

print()
print("=" * 66)
print("3) PyraCanny vs single-scale Canny (the actual reason to have it)")
print("=" * 66)
single = cv2.Canny(img, 64, 128)
pyra = out[..., 0]
check(np.count_nonzero(pyra) > np.count_nonzero(single),
      f"PyraCanny keeps more structure ({np.count_nonzero(pyra)} vs "
      f"{np.count_nonzero(single)} edge pixels)")

print()
print("=" * 66)
print("4) PIL wrapper")
print("=" * 66)
from PIL import Image

pil = Image.fromarray(img)
edge_pil = cu.pyracanny_from_pil(pil, 64, 128)
check(edge_pil.size == pil.size, f"same size ({edge_pil.size})")
check(edge_pil.mode == "RGB", f"RGB ({edge_pil.mode})")
check(np.array_equal(np.asarray(edge_pil), out), "PIL path matches the array path")

print()
print("=" * 66)
print("5) threshold mapping (0..1 UI -> 0..255 OpenCV)")
print("=" * 66)
for ul, uh, want in [
    (0.25, 0.5, (64, 128)),
    (0.0, 1.0, (0, 255)),
    (0.5, 0.5, (128, 129)),
    (0.8, 0.2, (204, 205)),
    (2.0, -1.0, (254, 255)),   # both clamp to the top, then reopened
    (1.0, 1.0, (254, 255)),
    (float("nan"), 0.5, (64, 128)),   # NaN falls back to the default low
]:
    got = cu.threshold_from_unit(ul, uh)
    check(got == want, f"({ul}, {uh}) -> {got} (expected {want})")
    check(0 <= got[0] < got[1] <= 255, f"   valid window {got}")

print()
print("=" * 66)
print("6) tiny images don't crash")
print("=" * 66)
for size in [(8, 8), (17, 33), (1, 1)]:
    try:
        r = cu.canny_pyramid(np.full((*size, 3), 128, np.uint8), 64, 128)
        check(r.shape == (*size, 3), f"{size} handled -> {r.shape}")
    except Exception as exc:
        check(False, f"{size} raised {type(exc).__name__}: {exc}")

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PYRACANNY CHECKS PASSED")
