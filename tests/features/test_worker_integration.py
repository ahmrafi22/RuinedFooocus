"""Verify the async_worker post-processing integration without ComfyUI."""
import ast
import sys
import types
import traceback as tb
from pathlib import Path

REPO = Path(r"E:\go\face swap\RuinedFooocus")
WORKER = REPO / "modules" / "async_worker.py"
PIPELINE = REPO / "modules" / "image_pipeline.py"

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


src = WORKER.read_text(encoding="utf-8")
tree = ast.parse(src)

print("=" * 66)
print("1) the hook runs on each finished image, before it is saved")
print("=" * 66)
loop = None
for node in ast.walk(tree):
    if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "x":
        for stmt in node.body:
            if isinstance(stmt, ast.If) and "apply_postprocess" in ast.unparse(stmt):
                loop = node
check(loop is not None, "the loop over `imgs` calls apply_postprocess")

if loop:
    body = [ast.unparse(s) for s in loop.body]
    idx = next((i for i, s in enumerate(body) if "postprocess" in s), None)
    save = next((i for i, s in enumerate(body) if "local_temp_filename" in s), None)
    check(idx is not None and save is not None and idx < save,
          f"post-processing happens before the save ({idx} < {save})")

check("faceswap_enable" in src and "enhance_enable" in src,
      "both flags gate the post-process")

print()
print("=" * 66)
print("2) apply_face_swap still never loses an image")
print("=" * 66)
fn = next(n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "apply_face_swap")
ns = {"traceback": tb}
exec(compile(ast.get_source_segment(src, fn), "f", "exec"), ns)
apply_face_swap = ns["apply_face_swap"]

calls = []


class FakeFS:
    @staticmethod
    def swap_image(image, source, **kw):
        calls.append(kw)
        if kw.get("det_size") == 999:
            raise RuntimeError("boom")
        return None if source == "NOFACE" else "SWAPPED"


fake = types.ModuleType("modules.faceswap")
fake.swap_image = FakeFS.swap_image
ns["modules"] = types.SimpleNamespace(faceswap=fake)

gen = {"faceswap_source": "SRC", "faceswap_det_size": 640}
check(apply_face_swap("IMG", gen) == "SWAPPED", "normal swap returns the result")
check(apply_face_swap("IMG", {"faceswap_source": None}) == "IMG", "no source -> original")
check(apply_face_swap("IMG", {**gen, "faceswap_source": "NOFACE"}) == "IMG",
      "no face detected -> original")
check(apply_face_swap("IMG", {**gen, "faceswap_det_size": 999}) == "IMG",
      "exception -> original")

print()
print("=" * 66)
print("3) apply_postprocess ordering: upscale first, swap last")
print("=" * 66)
fn2 = next(n for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "apply_postprocess")
ns2 = {"traceback": tb, "apply_face_swap": apply_face_swap}
exec(compile(ast.get_source_segment(src, fn2), "f2", "exec"), ns2)
apply_postprocess = ns2["apply_postprocess"]

order = []


class FakeEnh:
    DEFAULT_UPSCALER = "4x-UltraSharp.pth"

    @staticmethod
    def enhance_image(image, upscale_model_name=None, upscale_factor_long_edge=0,
                      swap_target=None, swap_kwargs=None):
        order.append(("enhance", swap_target))
        return "ENHANCED"


ns2["modules"] = types.SimpleNamespace(
    faceswap=types.SimpleNamespace(swap_image=lambda *a, **k: order.append(("swap",)) or "SWAPPED"),
    enhance=FakeEnh,
)
# apply_face_swap resolved `modules.faceswap` when it was compiled, so rebind
# it to the tracking stub as well.
ns["modules"] = ns2["modules"]

# enhance off -> plain swap
order.clear()
r = apply_postprocess("IMG", {"faceswap_source": "SRC", "faceswap_enable": True})
check(r == "SWAPPED", "enhance off -> the plain swap runs")
check(order == [("swap",)], f"only the swap ran ({order})")

# enhance on with a swap target -> enhance is told to swap at the end
order.clear()
r = apply_postprocess("IMG", {"faceswap_source": "SRC", "faceswap_enable": True,
                              "enhance_enable": True, "enhance_max_edge": 2048})
check(r == "ENHANCED", "enhance on -> enhance_image handles it")
check(order == [("enhance", "SRC")],
      f"the swap is handed to enhance (so it runs AFTER upscaling) ({order})")

# enhance on, swap off -> upscale only
order.clear()
r = apply_postprocess("IMG", {"enhance_enable": True})
check(r == "ENHANCED", "upscale only works")
check(order == [("enhance", None)], f"no swap target passed ({order})")

print()
print("=" * 66)
print("4) enhance failure falls back to a plain swap")
print("=" * 66)
ns3 = {"traceback": tb, "apply_face_swap": apply_face_swap}
exec(compile(ast.get_source_segment(src, fn2), "f3", "exec"), ns3)


class BoomEnh:
    DEFAULT_UPSCALER = "x"

    @staticmethod
    def enhance_image(*a, **k):
        raise RuntimeError("upscale exploded")


ns3["modules"] = types.SimpleNamespace(
    faceswap=types.SimpleNamespace(swap_image=lambda *a, **k: "SWAPPED"),
    enhance=BoomEnh,
)
r = ns3["apply_postprocess"]("IMG", {"faceswap_source": "SRC",
                                     "faceswap_enable": True,
                                     "enhance_enable": True})
check(r == "SWAPPED", "a broken upscaler still yields the swapped image")

print()
print("=" * 66)
print("5) a faceswap Cheat Code is never fed to the sampler")
print("=" * 66)
psrc = PIPELINE.read_text(encoding="utf-8")
check('controlnet["type"].lower() != "faceswap"' in psrc,
      "image_pipeline skips ControlNet for type faceswap")

print()
print("=" * 66)
print("6) pyracanny is wired into the pipeline")
print("=" * 66)
check('case "pyracanny":' in psrc, "the pipeline has a pyracanny branch")
check("pyracanny_from_pil" in psrc, "it calls the PyraCanny implementation")
check("threshold_from_unit" in psrc, "thresholds are mapped 0..1 -> 0..255")

cnsrc = (REPO / "modules" / "controlnet.py").read_text(encoding="utf-8")
check('"pyracanny"' in cnsrc, "pyracanny is a registered Cheat Code type")

uisrc = (REPO / "modules" / "ui" / "ui_controlnet.py").read_text(encoding="utf-8")
check('"pyracanny": [True, True, True, True, True, False]' in uisrc,
      "the UI shows the edge sliders for pyracanny")

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("WORKER INTEGRATION CHECKS PASSED")
