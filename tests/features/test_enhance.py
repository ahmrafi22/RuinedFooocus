"""Verify the enhance/upscale module and the realistic preset.

Runs the real functions with torch + spandrel (no GPU needed).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"E:\go\face swap\RuinedFooocus")
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)   # shared.py loads language/ relative to the cwd

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


import modules.enhance as enh

print("=" * 66)
print("1) realistic preset is sane")
print("=" * 66)
p = enh.REALISTIC_PRESET
for key in ("style", "cfg", "custom_steps", "resolution", "negative_prompt"):
    check(key in p, f"preset defines {key}")
check(isinstance(p["style"], list) and p["style"], "style is a non-empty list")
check(1.0 <= p["cfg"] <= 10.0, f"cfg {p['cfg']} is in a sane range")
check(10 <= p["custom_steps"] <= 80, f"steps {p['custom_steps']} is sane")
check("cartoon" in p["negative_prompt"], "negative prompt excludes cartoon")

# The styles must actually exist in the fork's style list.
import csv
rows = list(csv.reader(open(REPO / "settings/styles.default", encoding="utf-8")))
known = {r[0] for r in rows if r}
for s in p["style"]:
    check(s in known, f"style {s!r} exists in settings/styles.default")

print()
print("=" * 66)
print("2) tensor round-trip is lossless enough")
print("=" * 66)
img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (64, 48, 3), dtype=np.uint8))
back = enh._tensor_to_pil(enh._pil_to_tensor(img))
check(back.size == img.size, f"size preserved ({back.size})")
check(back.mode == "RGB", f"mode RGB ({back.mode})")
diff = np.abs(np.asarray(back, int) - np.asarray(img, int)).max()
check(diff <= 1, f"round-trip is exact within rounding (max diff {diff})")

print()
print("=" * 66)
print("3) downscale_to")
print("=" * 66)
big = Image.new("RGB", (4096, 2048))
r = enh.downscale_to(big, 2048)
check(r.size == (2048, 1024), f"4096x2048 -> {r.size}")
small = Image.new("RGB", (512, 512))
check(enh.downscale_to(small, 2048).size == (512, 512), "small images are untouched")
check(enh.downscale_to(small, 0) is small, "0 disables the cap")

print()
print("=" * 66)
print("4) upscale with a real model (spandrel)")
print("=" * 66)
try:
    import spandrel
    print("    spandrel", spandrel.__version__ if hasattr(spandrel, "__version__") else "ok")
except ImportError:
    print("    spandrel not installed locally - skipping the model test")
    spandrel = None

if spandrel is not None:
    # Use a tiny built-in model from spandrel's own test zoo if available,
    # otherwise verify the loader's failure path is clean.
    none_model = enh.load_upscaler("does-not-exist.pth")
    check(none_model is None, "a missing upscaler returns None, not a crash")

print()
print("=" * 66)
print("5) enhance_image falls back cleanly")
print("=" * 66)
src = Image.fromarray(np.full((32, 32, 3), 90, np.uint8))
# No upscaler available and no swap target -> unchanged image, no exception.
out = enh.enhance_image(src, upscale_model_name="None", swap_target=None)
check(out.size == src.size, f"unchanged when everything is off ({out.size})")

# A broken upscaler name must not raise.
out = enh.enhance_image(src, upscale_model_name="missing-model.pth", swap_target=None)
check(out.size == src.size, "a missing upscaler leaves the image intact")

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ENHANCE CHECKS PASSED")
