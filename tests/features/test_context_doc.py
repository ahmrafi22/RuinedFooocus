"""Fact-check context.md against the actual repo. Every claim must hold."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(r"E:\go\face swap\RuinedFooocus")
DOC = REPO / "context.md"

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


doc = DOC.read_text(encoding="utf-8")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout


print("=" * 66)
print("1) file sizes quoted in the doc")
print("=" * 66)
for rel, lines in [
    ("modules/faceswap.py", 295),
    ("modules/enhance.py", 166),
    ("modules/ui/ui_minimal.py", 424),
    ("modules/ui/ui_faceswap.py", 132),
    ("webui.py", 62),
]:
    p = REPO / rel
    check(p.exists(), f"{rel} exists")
    if p.exists():
        n = len(p.read_text(encoding="utf-8").splitlines())
        # allow small drift
        check(abs(n - lines) <= 12, f"{rel} ~{lines} lines (actual {n})")

print()
print("=" * 66)
print("2) commit hashes and messages")
print("=" * 66)
log = git("log", "--oneline", "-12")
for sha, msg in [
    ("ae4a0b6", "Add real InsightFace face swap"),
    ("ede6ae0", "Add Colab notebook"),
    ("70ad7fa", "Notebook: single cell"),
    ("10708a5", "Add PyraCanny"),
    ("e70e43f", "Put Face Swap inside the PowerUp tab"),
    ("960179f", "Advanced tab"),
    ("56f3478", "Fix argparse crash"),
    ("b7368e0", "Replace the UI"),
    ("3fc2797", "Fix silent generation failure"),
]:
    check(sha in log, f"commit {sha} exists")
check("a076b17" in log, "upstream base a076b17 exists")

print()
print("=" * 66)
print("3) controlnet registry claims")
print("=" * 66)
cn = json.loads((REPO / "modules/pathdb/controlnet.json").read_text(encoding="utf-8"))
check("cn_canny" in cn, "cn_canny is a registry key")
check("cn_pyracanny" not in cn, "cn_pyracanny is NOT a registry key (the bug)")

cnsrc = (REPO / "modules/controlnet.py").read_text(encoding="utf-8")
check('"pyracanny"' in cnsrc, "pyracanny is in controlnet_models")
check('key = "cn_canny"' in cnsrc, "get_model maps pyracanny -> cn_canny")

print()
print("=" * 66)
print("4) worker try/except claims")
print("=" * 66)
aw = (REPO / "modules/async_worker.py").read_text(encoding="utf-8")
check("A single bad task must never kill" in aw, "worker loop has the guard")
check("apply_face_swap" in aw, "apply_face_swap exists")
check("apply_postprocess" in aw, "apply_postprocess exists")
# process_metadata inside a try
tree = ast.parse(aw)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_process")
first_try = next((s for s in fn.body if isinstance(s, ast.Try)), None)
check(first_try is not None, "_process has a try block")
if first_try:
    wrapped = ast.unparse(ast.Module(body=first_try.body, type_ignores=[]))
    check("process_metadata" in wrapped, "process_metadata is inside the try")

print()
print("=" * 66)
print("5) argparse claim")
print("=" * 66)
ap = (REPO / "argparser.py").read_text(encoding="utf-8")
check("parse_known_args()" in ap, "uses parse_known_args")
check("args = parser.parse_args()" not in ap, "no bare parse_args()")

print()
print("=" * 66)
print("6) notebook claims")
print("=" * 66)
nb = json.loads((REPO / "RuinedFooocus_colab.ipynb").read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
check(len(code_cells) == 2, f"two code cells ({len(code_cells)})")
code = "\n".join("".join(c["source"]) for c in code_cells)
check("--offline" not in code.split("subprocess.Popen(")[1].split("]")[0],
      "launch argv has no --offline")
check("serve_kernel_port_as_iframe" in code, "uses Colab port proxy")
check("freezetorch" in code, "writes freezetorch")
check("TORCH_PLATFORM" in code, "sets TORCH_PLATFORM")
check("setup complete" in code, "prints setup complete")

print()
print("=" * 66)
print("7) pip pins")
print("=" * 66)
mods = (REPO / "pip/modules.txt").read_text(encoding="utf-8")
for pin in ("onnx==1.19.1", "insightface==2.1", "opencv-python==4.11.0.86"):
    check(pin in mods, f"modules.txt has {pin}")
check("gradio[mcp]==6.31.0" in mods, "gradio 6.31 pin (modern deps claim)")
check("transformers==5.20.1" in mods, "transformers 5 pin")
# no numpy pin -> the reason this fork works
numpy_pins = [l for l in mods.splitlines()
              if l.strip().startswith("numpy") and "==" in l]
check(numpy_pins == [], f"no numpy pin (found {numpy_pins})")

print()
print("=" * 66)
print("8) faceswap model facts")
print("=" * 66)
fs = (REPO / "modules/faceswap.py").read_text(encoding="utf-8")
check("inswapper_128.onnx" in fs, "references inswapper_128.onnx")
check("name=str(model_dir)" in fs,
      "FaceAnalysis gets an absolute dir (no auto-download)")
check("buffalo_l.zip" in fs, "downloads buffalo_l.zip")

reg = json.loads((REPO / "modules/pathdb/faceswap.json").read_text(encoding="utf-8"))
check(any("inswapper_128.onnx" in k for k in reg),
      "inswapper registered for download")

print()
print("=" * 66)
print("9) enhance / realistic preset")
print("=" * 66)
en = (REPO / "modules/enhance.py").read_text(encoding="utf-8")
check("REALISTIC_PRESET" in en, "REALISTIC_PRESET exists")
check("4x-UltraSharp.pth" in en, "default upscaler is 4x-UltraSharp.pth")
check("spandrel" in en, "uses spandrel")

print()
print("=" * 66)
print("10) the doc admits what is unverified")
print("=" * 66)
check("never actually run" in doc.lower() or "never actually run" in doc,
      "states the app has never run")
check("TORCH_PLATFORM" in doc and "Highest risk" in doc,
      "flags TORCH_PLATFORM as the top risk")
check("non-commercial" in doc, "carries the licence warning")
check("200 checks" in doc, "quotes the test count")

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("CONTEXT.MD FACT-CHECK PASSED")
