"""Verify the two-cell notebook and the argparse fix."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"E:\go\face swap\RuinedFooocus")
NB = REPO / "RuinedFooocus_colab.ipynb"

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


nb = json.loads(NB.read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
code = "\n".join("".join(c["source"]) for c in code_cells)

print("=" * 66)
print("1) two code cells: setup, then show")
print("=" * 66)
check(len(code_cells) == 2, f"exactly two code cells ({len(code_cells)})")
setup = "".join(code_cells[0]["source"])
show = "".join(code_cells[1]["source"])
check("Setup" in setup, "the first cell is setup")
check("Show the UI" in show, "the second cell shows the UI")
check("serve_kernel_port_as_iframe" in show, "the UI is embedded in the second cell")
check("serve_kernel_port_as_iframe" not in setup,
      "setup does NOT embed the UI (that was the confusing part)")

print()
print("=" * 66)
print("2) setup downloads everything before finishing")
print("=" * 66)
for need, why in [
    ("git\", \"clone", "clones the fork"),
    ("launch.py", "installs + starts"),
    ("sd_xl_base_1.0", "downloads the checkpoint"),
    ("faceswap.get_swapper", "downloads inswapper_128"),
    ("faceswap.get_analyzer", "downloads buffalo_l"),
    ("proc.poll()", "detects a crash"),
    ("tail(", "prints the real error"),
]:
    check(need in setup, why)

check("setup complete" in setup, "prints an explicit 'setup complete'")

print()
print("=" * 66)
print("3) the show cell restarts the server if needed")
print("=" * 66)
check("if not up():" in show, "checks whether the server is alive")
check("subprocess.Popen" in show, "can restart it")
check("launch.py" in show, "restart uses launch.py")

print()
print("=" * 66)
print("4) no NameError on the #@param variables")
print("=" * 66)
for var in ("FORCE_REINSTALL", "SKIP_CHECKPOINT"):
    d = setup.find(f"{var} = False")
    u = setup.find(var, d + 1)
    check(d != -1, f"{var} is assigned")
    check(d != -1 and (u == -1 or u > d), f"{var} is assigned before use")

print()
print("=" * 66)
print("5) both cells compile")
print("=" * 66)
for i, c in enumerate(code_cells):
    try:
        compile("".join(c["source"]), f"<cell {i}>", "exec")
        check(True, f"cell {i} compiles")
    except SyntaxError as e:
        check(False, f"cell {i}: {e}")

print()
print("=" * 66)
print("6) argparser survives a Jupyter kernel command line")
print("=" * 66)
# This is the exact failure the user hit: kernel argv with -f <json>.
probe = r'''
import sys
sys.argv = ["/usr/local/lib/python3.13/dist-packages/ipykernel_launcher.py",
            "-f", "/root/.local/share/jupyter/runtime/kernel-abc.json"]
sys.path.insert(0, %r)
import argparser
print("IMPORTED", argparser.args.port, argparser.args.listen)
''' % str(REPO)

r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
print("    exit:", r.returncode)
check(r.returncode == 0,
      f"importing argparser with kernel argv does not exit (got {r.returncode})")
check("IMPORTED" in r.stdout, "it imported successfully")
check("unrecognized arguments" not in r.stderr,
      "no 'unrecognized arguments' error")

print()
print("=" * 66)
print("7) real CLI flags still parse")
print("=" * 66)
probe2 = r'''
import sys
sys.argv = ["launch.py", "--nobrowser", "--listen", "127.0.0.1", "--port", "7865"]
sys.path.insert(0, %r)
import argparser
a = argparser.args
print("RESULT", a.port, a.listen, a.nobrowser)
''' % str(REPO)

r2 = subprocess.run([sys.executable, "-c", probe2], capture_output=True, text=True)
check(r2.returncode == 0, "CLI parse exits 0")
check("RESULT 7865 127.0.0.1 True" in r2.stdout,
      f"flags are honoured ({r2.stdout.strip().splitlines()[-1] if r2.stdout.strip() else 'none'})")

print()
print("=" * 66)
print("8) argparser uses parse_known_args, not parse_args")
print("=" * 66)
src = (REPO / "argparser.py").read_text(encoding="utf-8")
check("parse_known_args()" in src, "uses parse_known_args()")
check("args = parser.parse_args()" not in src,
      "no bare parse_args() that would exit on unknown flags")

print()
print("=" * 66)
if fails:
    print(f"{len(fails)} FAILURE(S)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("NOTEBOOK + ARGPARSE CHECKS PASSED")
