# context.md â€” RuinedFooocus + Face Swap + PyraCanny

Everything an agent needs to continue this work without re-doing the research.
Last updated: 2026-11-01.

---

## 1. What the user wants

A Colab notebook that runs a **Fooocus-like** image generator **inside the notebook** (no `gradio.live`, no public URL), with exactly three things in the UI:

1. **Prompt â†’ image**
2. **Pose reference (PyraCanny) + Face reference (real face swap) â†’ image**
3. **Upload an image â†’ upscale it**

Plus a **realistic** photographic look, not illustration.

The user has been burned by repeated silent failures and explicitly dislikes:
- being told to "check the logs" when the logs are invisible
- UI elements that cannot be found
- anything that requires them to debug

**Rule for future work: verify by executing real code paths, never by reading source and assuming.**

---

## 2. Repository

| | |
|---|---|
| Fork (deliverable) | `https://github.com/ahmrafi22/RuinedFooocus` |
| Upstream | `https://github.com/runew0lf/RuinedFooocus` |
| Upstream base commit | `a076b17` ("Revise installation section in README for 26.10.31") |
| Local clone | `E:\go\face swap\RuinedFooocus` |
| Notebook | `RuinedFooocus_colab.ipynb` in the fork root |
| Tests | `tests/features/` (200 checks) |

Commits on top of upstream (oldest first):

```
ae4a0b6  Add real InsightFace face swap
ede6ae0  Add Colab notebook that runs the UI inline
70ad7fa  Notebook: single cell, and stop passing --offline
10708a5  Add PyraCanny, enhance/upscale and a realistic preset
e70e43f  Put Face Swap inside the PowerUp tab, fix notebook NameError
960179f  Advanced tab: pose + face references drive one generation
56f3478  Fix argparse crash under a notebook kernel; split the notebook in two
b7368e0  Replace the UI with three things: generate, references, upscale
3fc2797  Fix silent generation failure and the prompt requirement
```

---

## 3. Why RuinedFooocus (research that must not be repeated)

The user originally linked `lllyasviel/Fooocus`. **It cannot run on Colab and no fork of it can.**

```
Colab today          -> Python 3.13.15   (upgraded 2026-09-16)
lllyasviel/Fooocus   -> gradio==3.41.2 -> requires numpy~=1.0
numpy 1.26.4         -> cp312 wheels only, NO cp313 wheel
```

So every traditional Fooocus fork needs the **2026.07 runtime (Python 3.12)** pinned manually, which is a year-limited past runtime.

Fork survey (all still pin `gradio 3.41.2` + `numpy 1.26.4`):

| Fork | Stars | Last real commit | Modern deps? |
|---|---|---|---|
| **runew0lf/RuinedFooocus** | 719 | 2026-10-31 | **YES** |
| DavidDragonsage/FooocusPlus | 134 | 2026-10-15 | no |
| shaitanzx/Fooocus_extend | 88 | 2026-10-31 | no |
| mikecastrodemaria/Fooocus2026 | 21 | 2026-10-31 | no |
| MoonRide303/Fooocus-MRE | 978 | 2026-01-16 | no |

`lllyasviel/Fooocus` last real commit **2025-01-24** (v2.5.5, unmaintained).
`mashb1t/Fooocus` (the old recommended fork) last real commit **2024-09-17**.

**RuinedFooocus is the only fork that escaped the numpy trap**: `pip/modules.txt` has no numpy pin, uses `gradio[mcp]==6.31.0`, `transformers==5.20.1`, `pydantic==2.12.5`, and carries explicit `onnxruntime==...; python_version >= "3.13"` branches.

It also **bundles ComfyUI** (clones it at a pinned hash) and imports it directly (`comfy.model_base`, `nodes.CLIPTextEncode`, `comfy.samplers.KSampler`) rather than using the old `ldm_patched`. This is what made real face swap and PyraCanny tractable.

---

## 4. Architecture

```
launch.py                          # upstream: installs deps, clones ComfyUI, then imports webui
  â””â”€â”€ webui.py                     # REWRITTEN: 62 lines, builds the minimal UI
        â””â”€â”€ modules/ui/ui_minimal.py
              â”œâ”€â”€ generate()       # prompt + refs -> worker.add_task -> image
              â””â”€â”€ upscale_image()  # modules/enhance.py

modules/async_worker.py            # worker thread; calls apply_postprocess per image
  â”œâ”€â”€ apply_face_swap()            # modules/faceswap.py
  â””â”€â”€ apply_postprocess()          # enhance first, swap LAST

modules/image_pipeline.py          # ComfyUI-based sampling; canny/pyracanny branches
modules/controlnet.py              # Cheat Code registry
modules/canny_utils.py             # PyraCanny (cv2 port of Fooocus's canny_pyramid)
```

Generation flow:

```
prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
pose image â†’ PyraCanny (cv2 pyramid) â†’ ControlNet â”€â”€â”¤â†’ KSampler â†’ VAE decode
                                                     â”‚        â”‚
face image â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜        â”‚
                                                              â–¼
                                          ReActor-style inswapper swap (modules/faceswap.py)
                                                              â–¼
                                          optional upscale (modules/enhance.py, spandrel)
                                                              â–¼
                                                           image
```

---

## 5. Files changed, and why

### New

| File | Lines | Purpose |
|---|---|---|
| `modules/faceswap.py` | 295 | InsightFace `inswapper_128` + buffalo_l detector. Self-downloading, cached, thread-safe. `swap_image()`, `status()`, `unload()`. |
| `modules/enhance.py` | 166 | `REALISTIC_PRESET`, spandrel upscaler loading, `upscale()`, `downscale_to()`, `enhance_image()` (upscale then swap last). |
| `modules/ui/ui_minimal.py` | 424 | The whole UI: generate, references, upscale. |
| `modules/ui/ui_faceswap.py` | 132 | Face-swap option controls (used by the old Advanced tab; **now unused by `webui.py`** but still imported by `ui_controlnet.py`). |
| `modules/canny_utils.py` | +133 | PyraCanny port, plus `threshold_from_unit()`. |
| `modules/pathdb/faceswap.json` | +5 | Registers `inswapper_128.onnx` for auto-download. |
| `RuinedFooocus_colab.ipynb` | â€” | Two cells: setup, then show UI. |

### Modified

| File | Change |
|---|---|
| `webui.py` | **Replaced.** 1503 â†’ 62 lines. Builds the minimal UI, keeps the launch contract (`shared.gradio_root`, `prevent_thread_lock=True`, trailing `while True`). |
| `modules/async_worker.py` | `apply_face_swap()`, `apply_postprocess()`, hook in the per-image save loop, try/except around `handler(task)`, try/except around `process_metadata`/`pipelines.update`. |
| `modules/image_pipeline.py` | `case "pyracanny"` branch; `pose_enable` auto-selects PyraCanny; skips ControlNet when type is `faceswap`. |
| `modules/controlnet.py` | `"pyracanny"` in `controlnet_models`; `get_model()` maps `pyracanny â†’ cn_canny`. |
| `argparser.py` | `parse_args()` â†’ `parse_known_args()` (Colab kernel argv crash). |
| `pip/modules.txt` | `+ onnx==1.19.1`, `+ insightface==2.1`, `+ opencv-python==4.11.0.86`. |

---

## 6. The bugs that actually bit (do not reintroduce)

### 6.1 `--offline` skips the entire pip install
In `launch.py`, `if offline:` guards **all** of `prepare_environment`, including
`pip install -r pip/modules.txt`. The notebook must **never** pass `--offline`.

### 6.2 argparse dies on a Jupyter kernel's argv
`argparser.py` called `parse_args()` at import. In a notebook `sys.argv` is
`['ipykernel_launcher.py', '-f', '.../kernel-*.json']` â†’ `SystemExit: 2`.
Fixed with `parse_known_args()`.

### 6.3 `pyracanny` had no ControlNet model
`get_model()` derived `f"cn_{type}"` â†’ `cn_pyracanny`, which is **not** a
registry key. No ControlNet loaded, pose reference silently ignored.
Fixed: `pyracanny â†’ cn_canny` (same model, different edge extraction).

### 6.4 One failure killed the worker thread forever
`async_worker.worker()` had `handler(task)` with **no try/except** inside its
`while True`. Any exception killed the thread, so every later Generate returned
nothing, with the traceback going to a stdout nobody could see.
Fixed: catch, print, and append an empty `"results"` so the caller unblocks.

### 6.5 `_process` could die before the pipeline ran
`process_metadata()` and `pipelines.update()` were outside any try.
Now wrapped.

### 6.6 Upscaler dropdown crash
`4x-UltraSharp.pth is not in the list of choices: ['None']` â€” the list came from
a **disk scan** (empty on fresh install) while the default was a name from the
**download registry**. Fixed: `_upscaler_list()` reads the registry (35 entries).

### 6.7 Colab `#@param` variables are undefined on kernel execution
`SKIP_CHECKPOINT` / `RESTART_APP` raised `NameError`. Now assigned explicitly.

### 6.8 UI discoverability
A top-level tab was invisible because the tab strip scrolls. Tried PowerUp, then
Advanced, then dropped tabs entirely in favour of the minimal UI.

---

## 7. Verified numbers (do not re-derive)

### PyraCanny
Byte-identical to `comfyui_controlnet_aux`'s `PyraCannyPreprocessor` and to
Fooocus's `canny_pyramid`, across 3 images Ã— 3 threshold pairs = **9/9 exact**.

```
9 scales: 0.2 .. 1.0
acc = acc*0.75 + edge*0.25
sum channels -> percentile normalise (1, 99) -> 0..255
```

PyraCanny keeps **48197** edge pixels vs **6270** for single-scale Canny (7.7Ã—).

Threshold mapping: UI sliders are `0..1` (ComfyUI convention), `cv2.Canny` wants
`0..255`. `threshold_from_unit()` clamps, and reopens an invalid window
(`low >= 255 â†’ (254, 255)`).

### Face swap
Real swap verified on real faces with real models:
- `inswapper_128.onnx` 554 MB, input 128Ã—128
- buffalo_l unpacks to exactly 5 files; top-level `buffalo_l/` stripped
- output keeps target size, is not blank, identity visibly transferred

`FaceAnalysis(name=<absolute dir>)` â€” passing an **absolute directory** makes it
read `.onnx` files directly instead of looking in `<root>/models/<name>` and
downloading. This is deliberate.

### Upscale
`4x-UltraSharp.pth` (67 MB) auto-downloads, loads via spandrel, real
`128Ã—128 â†’ 512Ã—512`.

### Test suite â€” 200 checks, 0 failures

Location: **`tests/features/`** in the repo. Run any file directly:

```bash
cd "E:\go\face swap\RuinedFooocus\tests\features"
python test_pyracanny.py
```

| File | Checks | Covers |
|---|---|---|
| `test_pyracanny.py` | 35 | byte-identical to upstream reference, thresholds, tiny images |
| `test_enhance.py` | 20 | realistic preset, tensor round-trip, downscale, fallbacks |
| `test_upscale_real.py` | 7 | real ESRGAN load + 4x upscale |
| `test_upscaler_resolve.py` | 5 | dropdown name â†’ download â†’ load (the crash) |
| `test_minimal_ui.py` | 42 | UI builds, only 3 things, no leftover controls |
| `test_generate_logic.py` | 18 | prompt/refs guards, pyracanny selection, payload |
| `test_silent_failure.py` | 29 | the three silent-failure bugs stay fixed |
| `test_worker_integration.py` | 20 | hook position, failure paths, ordering |
| `test_two_cell.py` | 29 | notebook shape + argparse regression |
| `test_context_doc.py` | 54 | fact-checks this document against the repo |

Note: several tests import `modules.*`, which needs `shared` (cwd must be the
repo root) and stubs for ComfyUI-dependent modules. `test_minimal_ui.py` and
`test_advanced_tab.py`-style tests show the stubbing pattern.

---

## 8. Notebook design

Two cells, deliberately.

**Cell 1 â€” Setup:** checks GPU â†’ clones/pulls fork â†’ runs `launch.py` in the
background (**without `--offline`**) â†’ streams the log â†’ waits for HTTP 200 on
`127.0.0.1:7865` â†’ downloads SDXL base (6.9 GB) â†’ downloads swap models (843 MB)
â†’ prints `setup complete`.

**Cell 2 â€” Show the UI:** `google.colab.output.serve_kernel_port_as_iframe(7865)`,
and restarts the server if it is gone.

Why a background process: `launch.py` ends by importing `webui.py`, which ends in
`while True: time.sleep(100)` â€” it never returns, so it cannot run inline.

---

## 9. Environment facts

- Colab runtime: **Python 3.13.15**, Ubuntu 24.04. Leave runtime version on **Latest**.
- Colab ships its own CUDA torch. The notebook writes a `freezetorch` file so
  `torchruntime` does not reinstall it, and sets `TORCH_PLATFORM=cu124`.
- Disk: ~7.7 GB for models (6.9 GB checkpoint + 843 MB swap).
- `inswapper_128.onnx` is **InsightFace non-commercial research only**. Personal
  use fine; commercial use needs a licence.

---

## 10. What is NOT verified (read this before claiming success)

**The app has never actually run.** There is no GPU and no ComfyUI in the dev
environment. Everything above was verified at the logic level â€” real models, real
upscaling, real Gradio 6.31 blocks, real ONNX graphs, AST-level hook checks â€” but
**no end-to-end render has ever been observed.**

Specifically unverified:

1. **`TORCH_PLATFORM=cu124` vs Colab's preinstalled torch.** Highest risk. If the
   install step fails, this is the first suspect.
2. **xllamacpp install** from `xorbitsai.github.io` â€” unrelated to this work but
   in the startup path.
3. **Gradio 6 behind Colab's path proxy** â€” asset paths may not resolve; the
   iframe may need re-running.
4. **Whether `modules/async_worker.py`'s `gen_data` contract holds** for every
   pipeline (only the SDXL path was reasoned about).

---

## 11. Next steps

1. Run cell 1 in Colab; if it fails, the log tail is printed â€” fix that specific error.
2. Run cell 2; confirm the iframe renders.
3. Do one real generation with a pose + face reference.
4. Delete the now-unreachable UI modules. Verified against the current tree:
   - **Nothing imports them at all**: `ui_controlnet.py`, `ui_onebutton.py`,
     `ui_settings.py`, `ui_image_gallery.py`
   - **Only imported by other dead modules**: `ui_faceswap.py` (by
     `ui_controlnet.py`), `ui_evolve.py` and `ui_llama.py` (both by
     `ui_controlnet.py`)

   Delete `ui_controlnet.py` first, then the rest fall out. Re-grep before
   removing anything, and keep `modules/ui/__init__.py` if present.

---

## 12. Related earlier work (same session, separate deliverable)

`E:\go\face swap\modal_app.py` â€” a **ComfyUI + PyraCanny + ReActor + InstantID**
app for Modal (A100-40GB, $30/mo free credit). Fully tested (187 checks) but also
never run against a real GPU. This Colab notebook superseded it for the user's
use case. Kept for reference.

Key shared knowledge from that work:
- `comfyui_controlnet_aux`'s PyraCanny == Fooocus's `canny_pyramid` (same algorithm)
- Fooocus's "FaceSwap" is **IP-Adapter Plus Face**, not a real swap â€” confirmed in
  `modules/async_worker.py` lines 430â€“442
- InsightFace routes models by **graph shape**, not filename
- ReActor's `reactor_core` is self-contained (41 KB, no insightface import)

