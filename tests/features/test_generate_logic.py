"""Run the real generate() guard logic and the pyracanny selection.

Uses stubs for the heavy modules so the decision logic itself is exercised.
"""
import os
import sys
import types
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


# --- stub the heavy backend ------------------------------------------------
captured = {}


class FakeWorker:
    @staticmethod
    def add_task(gen_data):
        captured.update(gen_data)
        return 1

    @staticmethod
    def task_result(tid):
        return ("results", ["/tmp/out.png"])


class FakeEnhance:
    DEFAULT_UPSCALER = "4x-UltraSharp.pth"

    @staticmethod
    def load_upscaler(name):
        return object()

    @staticmethod
    def upscale(img, model=None):
        return img

    @staticmethod
    def downscale_to(img, edge):
        return img


sys.modules["modules.async_worker"] = FakeWorker
sys.modules["modules.enhance"] = FakeEnhance

import modules.ui.ui_minimal as ui

# point prune at a harmless dir
ui._prune_outputs = lambda keep=8: None


def run(prompt, pose=None, face=None, pyra=True, swap=True):
    captured.clear()
    try:
        ui.generate(prompt, "neg", "896x1152 (3:4)", "sd_xl_base_1.0_0.9vae.safetensors",
                    30, 5.0, -1, pose, face, pyra, swap,
                    progress=lambda *a, **k: None)
        return True, captured.copy()
    except Exception as exc:
        return False, exc


pose_img = Image.fromarray(np.full((400, 300, 3), 128, np.uint8))
face_img = Image.fromarray(np.full((200, 200, 3), 200, np.uint8))

print("=" * 66)
print("1) a prompt alone generates")
print("=" * 66)
ok, data = run("a woman in a red coat")
check(ok, "generate() succeeded")
check(data.get("cn_type") in (None, ""), "no controlnet without a pose ref")
check(data.get("faceswap_enable") is False, "no swap without a face ref")

print()
print("=" * 66)
print("2) references alone generate, without a prompt")
print("=" * 66)
ok, data = run("", pose=pose_img, face=face_img)
check(ok, "generate() succeeded with an empty prompt")
check(isinstance(data.get("prompt"), str) and len(data["prompt"]) > 10,
      "a default prompt was filled in")
check(data.get("cn_type") == "pyracanny", "PyraCanny is selected from the pose ref")
check(data.get("input_image") is not None, "the pose image was passed through")
check(data.get("faceswap_enable") is True, "face swap is enabled")
check(data.get("faceswap_source") is not None, "the face image was passed through")

print()
print("=" * 66)
print("3) nothing at all is still refused")
print("=" * 66)
ok, exc = run("")
check(not ok, "generate() refused")
check("Type a prompt" in str(exc), f"the message is clear ({exc})")

print()
print("=" * 66)
print("4) the reference toggles are honoured")
print("=" * 66)
ok, data = run("x", pose=pose_img, face=face_img, pyra=False, swap=False)
check(ok, "generated")
check(data.get("cn_type") in (None, ""), "PyraCanny off -> no controlnet")
check(data.get("faceswap_enable") is False, "swap off -> disabled")

print()
print("=" * 66)
print("5) the payload matches what the backend hard-reads")
print("=" * 66)
ok, data = run("x", pose=pose_img, face=face_img)
required = ["prompt", "negative", "style_selection", "seed", "base_model_name",
            "performance_selection", "custom_steps", "cfg", "sampler_name",
            "scheduler", "clip_skip", "aspect_ratios_selection",
            "custom_width", "custom_height", "image_number", "loras",
            "input_image", "main_view", "inpaint_toggle", "inpaint_view",
            "cn_selection", "cn_type"]
missing = [k for k in required if k not in data]
check(not missing, f"all required keys present" + (f" -- {missing}" if missing else ""))
check(data.get("performance_selection") == "Custom...",
      "CUSTOM_PERFORMANCE so steps/cfg are honoured")
check(data.get("aspect_ratios_selection") == "Custom...",
      "CUSTOM_RESOLUTION so width/height are honoured")
check(data.get("custom_width") == 896 and data.get("custom_height") == 1152,
      f"resolution applied ({data.get('custom_width')}x{data.get('custom_height')})")

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("GENERATE LOGIC CHECKS PASSED")
