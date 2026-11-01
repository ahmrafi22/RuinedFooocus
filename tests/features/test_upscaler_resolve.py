"""Prove the upscaler dropdown value can actually be resolved and used.

The crash was: the dropdown offered a name the app could not resolve.
This checks the name -> path -> loaded model chain for real.
"""
import os
import sys
from pathlib import Path

REPO = Path(r"E:\go\face swap\RuinedFooocus")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


import modules.enhance as enhance
from shared import path_manager

NAME = "4x-UltraSharp.pth"

print("=" * 66)
print("1) the dropdown name resolves to a real file (downloading if needed)")
print("=" * 66)
path = path_manager.get_file_path(NAME)
print("    resolved:", path)
check(path is not None, "get_file_path returned a path")
check(Path(path).exists(), "the file exists (downloaded on demand)")
if path:
    size = Path(path).stat().st_size
    print(f"    size: {size/1e6:.1f} MB")
    check(size > 1e6, "it is a real model, not an error page")

print()
print("=" * 66)
print("2) it loads through spandrel")
print("=" * 66)
model = enhance.load_upscaler(NAME)
check(model is not None, "load_upscaler succeeded")

print()
print("=" * 66)
print("3) a real upscale through the same code path the UI button uses")
print("=" * 66)
if model is not None:
    import numpy as np
    from PIL import Image

    src = Image.fromarray(np.random.default_rng(3).integers(0, 256, (128, 128, 3), np.uint8))
    out = enhance.upscale(src, model=model)
    check(out is not None, "upscale returned an image")
    if out is not None:
        check(out.size == (512, 512), f"4x ({src.size} -> {out.size})")
        capped = enhance.downscale_to(out, 2048)
        check(capped.size == (512, 512), "small result is not upscaled further")

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("UPSCALER RESOLUTION CHECKS PASSED")
