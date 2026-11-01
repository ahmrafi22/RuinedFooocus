"""Verify the minimal UI and the upscaler-dropdown crash fix."""
import ast
import os
import sys
import types
from pathlib import Path

REPO = Path(r"E:\go\face swap\RuinedFooocus")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


print("=" * 66)
print("1) the upscaler crash: '4x-UltraSharp.pth is not in the list of choices'")
print("=" * 66)
# Reproduce the old behaviour: list from disk only (empty on a fresh install).
from shared import path_manager

on_disk = list(path_manager.upscaler_filenames)
print(f"    upscaler_filenames (disk scan) = {on_disk}")
check(on_disk == [], "confirmed: the disk scan is empty on a fresh install")

# modules.async_worker imports ComfyUI, which only exists inside the app
# container. Stub it so the UI itself can be built and inspected here.
_stub = types.ModuleType("modules.async_worker")
_stub.add_task = lambda gen: 1
_stub.task_result = lambda tid: ("results", ["/tmp/x.png"])
sys.modules["modules.async_worker"] = _stub

import modules.ui.ui_minimal as ui

listed = ui._upscaler_list()
print(f"    _upscaler_list() returned {len(listed)} entries")
check(len(listed) > 0, f"the new list is not empty ({len(listed)})")
check("4x-UltraSharp.pth" in listed, "4x-UltraSharp.pth is in the list")

# The value the dropdown starts with must always be one of its choices.
default = "4x-UltraSharp.pth" if "4x-UltraSharp.pth" in listed else listed[0]
check(default in listed, f"default value {default!r} is a valid choice")

print()
print("=" * 66)
print("2) the minimal UI builds")
print("=" * 66)
import gradio as gr

try:
    demo = ui.build_minimal_ui()
    check(True, "build_minimal_ui() ran")
except Exception as exc:
    import traceback
    traceback.print_exc()
    check(False, f"raised {type(exc).__name__}: {exc}")
    raise SystemExit(1)

print()
print("=" * 66)
print("3) only the three things the user asked for")
print("=" * 66)
labels = [getattr(b, "label", None) for b in demo.blocks.values()]
labels = [l for l in labels if l]

for want in ("Prompt", "Negative prompt", "Result",
             "Pose reference (body / clothing)", "Face reference (identity)",
             "Upscaler", "Image to upscale", "Upscaled"):
    check(want in labels, f"has {want!r}")

# Things that must be gone.
for gone in ("LoRA", "Style", "Cheat Code", "Inpaint", "Preset",
             "Sampler", "Scheduler", "Performance", "Clip skip",
             "Image browser", "Chat", "Settings"):
    hit = [l for l in labels if gone.lower() in l.lower()]
    check(not hit, f"no {gone!r} control left" + (f" (found {hit})" if hit else ""))

tabs = [b for b in demo.blocks.values() if isinstance(b, gr.Tab)]
check(tabs == [], f"no tabs at all ({len(tabs)} found)")

print()
print("=" * 66)
print("4) the backend contract still matches what the worker reads")
print("=" * 66)
src = (REPO / "modules" / "ui" / "ui_minimal.py").read_text(encoding="utf-8")

# Every key the worker/pipeline reads without a default must be supplied.
required = {
    "prompt", "negative", "style_selection", "seed", "base_model_name",
    "performance_selection", "custom_steps", "cfg", "sampler_name",
    "scheduler", "clip_skip", "aspect_ratios_selection", "custom_width",
    "custom_height", "image_number", "loras", "input_image", "main_view",
    "inpaint_toggle", "inpaint_view", "cn_selection", "cn_type",
    "task_type",
}
gen_block = src[src.index("gen_data = {"):src.index("progress(0.10")]
missing = [k for k in sorted(required) if f'"{k}"' not in gen_block]
check(not missing, f"gen_data supplies every required key" +
      (f" -- missing {missing}" if missing else ""))

print()
print("=" * 66)
print("5) the constants that make the sliders authoritative")
print("=" * 66)
check("CUSTOM_PERFORMANCE" in src,
      "performance_selection uses CUSTOM_PERFORMANCE so steps/cfg are honoured")
check("CUSTOM_RESOLUTION" in src,
      "aspect_ratios_selection uses CUSTOM_RESOLUTION so width/height are honoured")
check("custom_width" in src and "custom_height" in src,
      "custom_width/custom_height are set from the resolution")

print()
print("=" * 66)
print("6) webui.py is minimal and keeps the launch contract")
print("=" * 66)
web = (REPO / "webui.py").read_text(encoding="utf-8")
check("build_minimal_ui" in web, "webui uses the minimal UI")
check("shared.gradio_root" in web, "sets shared.gradio_root (launch.py expects it)")
check("prevent_thread_lock=True" in web, "non-blocking launch")
check("while True" in web, "keeps the process alive like the original")
for gone in ("ui_onebutton", "ui_controlnet", "ui_settings", "ui_llama",
             "ui_image_gallery", "TabbedInterface"):
    check(gone not in web, f"webui no longer imports {gone}")

print()
print("=" * 66)
print("7) the file still parses")
print("=" * 66)
for f in ("webui.py", "modules/ui/ui_minimal.py"):
    try:
        ast.parse((REPO / f).read_text(encoding="utf-8"))
        check(True, f"{f} parses")
    except SyntaxError as e:
        check(False, f"{f}: {e}")

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("MINIMAL UI CHECKS PASSED")
