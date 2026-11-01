"""Prove the upscale path really runs, using a real ESRGAN model.

Downloads a small 4x model (a few MB), loads it through modules/enhance.py and
checks the output is genuinely 4x larger and not blank.
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"E:\go\face swap\RuinedFooocus")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


import modules.enhance as enh

# A tiny 4x ESRGAN-compatible model. Use the one the fork already registers
# if present, otherwise pull a small stand-in into a temp dir.
work = Path(os.environ.get("TEMP", "/tmp")) / "upscale_test"
work.mkdir(parents=True, exist_ok=True)
model_path = work / "4x-UltraSharp.pth"

if not model_path.exists():
    url = "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
    print("downloading 4x-UltraSharp.pth (small)...")
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as exc:
        print("download failed:", exc)

print("=" * 66)
print("1) the model loads through spandrel")
print("=" * 66)
check(model_path.exists(), f"model on disk ({model_path.stat().st_size/1e6:.1f} MB)"
      if model_path.exists() else "model missing")

from spandrel import ModelLoader
model = None
if model_path.exists():
    model = ModelLoader().load_from_file(str(model_path))
    check(model is not None, "spandrel loaded the model")
    print("    arch:", getattr(model, "architecture", None))

print()
print("=" * 66)
print("2) a real 4x upscale")
print("=" * 66)
if model is not None:
    # structured test image so edges are meaningful
    arr = np.zeros((96, 96, 3), np.uint8)
    arr[20:70, 20:70] = 200
    arr[35:55, 35:55] = 40
    img = Image.fromarray(arr)

    out = enh.upscale(img, model=model)
    check(out is not None, "upscale returned an image")
    if out is not None:
        check(out.size == (img.size[0] * 4, img.size[1] * 4),
              f"output is 4x ({img.size} -> {out.size})")
        a = np.asarray(out).astype(np.float32)
        check(a.mean() > 5, f"output is not blank (mean {a.mean():.1f})")
        check(a.std() > 5, f"output has detail (std {a.std():.1f})")
        out.save(work / "upscaled.png")

print()
print("=" * 66)
print("3) enhance_image end to end (upscale + cap)")
print("=" * 66)
if model is not None:
    import modules.enhance as e2

    # bypass load_upscaler so we use the model we already have
    original_loader = e2.load_upscaler
    e2.load_upscaler = lambda name=None: model
    try:
        img = Image.fromarray(np.full((64, 64, 3), 120, np.uint8))
        out = e2.enhance_image(img, upscale_model_name="4x-UltraSharp.pth",
                               upscale_factor_long_edge=128, swap_target=None)
        check(out.size == (128, 128),
              f"4x then capped to long edge 128 -> {out.size}")
    finally:
        e2.load_upscaler = original_loader

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("UPSCALE CHECKS PASSED")
