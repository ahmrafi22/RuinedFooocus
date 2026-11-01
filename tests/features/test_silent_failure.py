"""Verify the three bugs behind 'generation produced no image'.

1. controlnet.get_model('pyracanny') looked up 'cn_pyracanny', which does not
   exist, so no ControlNet loaded and the pose reference was ignored.
2. The worker's while-loop had no try/except, so one bad task killed the
   thread and every later Generate silently returned nothing.
3. _process could raise before the pipeline ran (process_metadata /
   pipelines.update), again killing the thread with no message.
"""
import ast
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


print("=" * 66)
print("1) pyracanny resolves to the canny ControlNet")
print("=" * 66)
from shared import path_manager

# The bug: the key was derived mechanically as f"cn_{type}".
buggy_key = "cn_pyracanny"
check(buggy_key not in path_manager.DOWNLOADABLE_FILES,
      "confirmed: 'cn_pyracanny' is not a registry key")

import modules.controlnet as controlnet

src = (REPO / "modules/controlnet.py").read_text(encoding="utf-8")
check('str(type).lower() == "pyracanny"' in src,
      "get_model special-cases pyracanny")
check('key = "cn_canny"' in src, "it maps pyracanny -> cn_canny")

# Simulate both lookups against the real registry.
def resolve(type_):
    key = "cn_canny" if str(type_).lower() == "pyracanny" else f"cn_{type_}"
    return key in path_manager.DOWNLOADABLE_FILES

check(resolve("pyracanny"), "pyracanny now resolves to a real entry")
check(resolve("canny"), "canny still resolves")
check(resolve("depth"), "depth still resolves")
check(not resolve("nonsense"), "an unknown type still resolves to nothing")

print()
print("=" * 66)
print("2) a failing task no longer kills the worker thread")
print("=" * 66)
w = (REPO / "modules/async_worker.py").read_text(encoding="utf-8")
tree = ast.parse(w)

loop = None
for node in ast.walk(tree):
    if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
        body = ast.unparse(node)
        # the worker's own loop is the one that pops from the buffer
        if "buffer.pop" in body:
            loop = node
check(loop is not None, "found the worker main loop (pops from buffer)")

if loop is not None:
    body = ast.unparse(loop)
    check("try" in body and "handler(task)" in body,
          "handler() is wrapped in a try")
    check("except Exception" in body, "it catches Exception")
    check("traceback.print_exc" in body, "it prints the traceback")
    check('"results"' in w, "it still answers the waiting caller")

print()
print("=" * 66)
print("3) _process cannot die silently before the pipeline runs")
print("=" * 66)
fn = next(n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "_process")
body_src = ast.unparse(fn)
check("process_metadata" in body_src, "still calls process_metadata")
# the first try must wrap process_metadata
first_try = next((s for s in fn.body if isinstance(s, ast.Try)), None)
check(first_try is not None, "there is a try block")
if first_try is not None:
    wrapped = ast.unparse(ast.Module(body=first_try.body, type_ignores=[]))
    check("process_metadata" in wrapped,
          "process_metadata is inside the try")
    check("pipelines.update" in wrapped,
          "pipelines.update is inside the try")
    check("except Exception" in ast.unparse(first_try),
          "the try has an except Exception")

print()
print("=" * 66)
print("4) the UI no longer demands a prompt when references are present")
print("=" * 66)
ui = (REPO / "modules/ui/ui_minimal.py").read_text(encoding="utf-8")
check("has_pose = pose_ref is not None" in ui, "checks for a pose reference")
check("has_face = face_ref is not None" in ui, "checks for a face reference")
check("Type a prompt, or add a pose / face reference." in ui,
      "the error names both options")
check("photograph of a person" in ui,
      "falls back to a sensible prompt instead of refusing")

print()
print("=" * 66)
print("5) the UI surfaces the real error instead of 'check the log'")
print("=" * 66)
check("_ERROR_LOG" in ui, "there is an error ring buffer")
check("_last_error()" in ui, "the failure path reads it")
gen_part = ui[ui.index("def generate("):ui.index("def upscale_image(")]
check("check the server log" not in gen_part,
      "the unhelpful message is gone from the generate path")
check("_capture_worker_errors" in ui, "worker prints are mirrored")

print()
print("=" * 66)
print("6) everything still compiles")
print("=" * 66)
for f in ("modules/async_worker.py", "modules/controlnet.py",
          "modules/ui/ui_minimal.py", "webui.py"):
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
print("SILENT-FAILURE CHECKS PASSED")
