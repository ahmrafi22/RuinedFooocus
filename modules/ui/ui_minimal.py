"""Minimal RuinedFooocus UI.

Three things only:

  1. Generate - prompt in, image out.
  2. Face + pose references - PyraCanny on the pose reference and an
     InsightFace swap of the face reference, applied in one generation.
  3. Enhance / upscale - take an image and upscale it.

Everything else in the original UI (LoRAs, styles, presets, cheat codes,
inpainting, image browser, chat, settings, model manager) is gone.
"""

import json
import os
import time
import traceback
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image

import modules.async_worker as worker
import modules.enhance as enhance
import modules.faceswap as faceswap
import shared
from shared import path_manager, performance_settings, resolution_settings, settings

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CUSTOM_PERFORMANCE = performance_settings.CUSTOM_PERFORMANCE
CUSTOM_RESOLUTION = resolution_settings.CUSTOM_RESOLUTION

RESOLUTIONS = [
    "896x1152 (3:4)",
    "832x1216 (2:3)",
    "1024x1024 (1:1)",
    "1152x896 (4:3)",
    "1216x832 (3:2)",
    "1344x768 (16:9)",
]
DEFAULT_RESOLUTION = "896x1152 (3:4)"

DEFAULT_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, 3d render, cgi, "
    "plastic skin, airbrushed, oversaturated, deformed, extra fingers, "
    "extra limbs, bad hands, blurry, lowres, watermark, text, jpeg artifacts"
)


def _checkpoint_list():
    """Checkpoints on disk, or the one the app knows how to download."""
    try:
        names = path_manager.get_model_filenames(
            path_manager.model_paths["modelfile_path"]
        )
    except Exception:
        names = []
    if names:
        return names
    return [settings.default_settings.get("base_model",
                                          "sd_xl_base_1.0_0.9vae.safetensors")]


def _upscaler_list():
    """Every upscaler spandrel can load, including ones not downloaded yet.

    The original code only listed files already on disk, which is why the
    dropdown came up as ['None'] on a fresh install and crashed with
    "4x-UltraSharp.pth is not in the list of choices".
    """
    names = set()

    try:
        for key, info in path_manager.DOWNLOADABLE_FILES.items():
            if info.get("path") == "path_upscalers":
                names.add(info.get("filename") or key)
    except Exception:
        pass

    try:
        names.update(path_manager.upscaler_filenames)
    except Exception:
        pass

    return sorted(n for n in names if n and n.lower().endswith((".pth", ".safetensors")))


def _save_temp(image, prefix):
    """Write a PIL image somewhere ComfyUI can read it, return the path."""
    folder = Path(path_manager.model_paths["temp_outputs_path"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    image.convert("RGB").save(path)
    return path


def _wait(task_id):
    """Block until the worker finishes, returning the output paths."""
    while True:
        flag, product = worker.task_result(task_id)
        if flag == "results":
            return product
        # "preview" products are progress updates; ignore them here.


def _prune_outputs(keep=8):
    """Delete all but the newest few generated files.

    Without this the outputs folder grows for the life of the session; every
    image is written there by async_worker before we ever see it.
    """
    try:
        folder = Path(path_manager.model_paths["temp_outputs_path"])
        files = sorted(
            (p for p in folder.glob("*.png") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. generate
# ---------------------------------------------------------------------------
def generate(
    prompt,
    negative,
    resolution,
    checkpoint,
    steps,
    cfg,
    seed,
    pose_ref,
    face_ref,
    use_pyracanny,
    use_faceswap,
    progress=gr.Progress(),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Type a prompt first.")

    progress(0.02, desc="preparing")
    tmp_files = []

    try:
        input_image = None
        if pose_ref is not None:
            input_image = Image.fromarray(pose_ref) if not isinstance(pose_ref, Image.Image) else pose_ref

        face_image = None
        if face_ref is not None:
            face_image = face_ref

        width, height = resolution_settings.aspect_ratios.get(
            resolution, (896, 1152)
        )

        gen_data = {
            "task_type": "process",
            "prompt": prompt,
            "negative": negative or "",
            "style_selection": [],
            "seed": int(seed) if seed not in (None, "") else -1,
            "base_model_name": checkpoint,
            "performance_selection": CUSTOM_PERFORMANCE,
            "custom_steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "dpmpp_2m_sde_gpu",
            "scheduler": "karras",
            "clip_skip": 1,
            "aspect_ratios_selection": CUSTOM_RESOLUTION,
            "custom_width": width,
            "custom_height": height,
            "image_number": 1,
            "loras": [],
            "input_image": input_image,
            "main_view": None,
            "inpaint_toggle": False,
            "inpaint_view": None,
            "cn_selection": None,
            "cn_type": "pyracanny" if (use_pyracanny and input_image) else None,
            "cn_edge_low": 0.25,
            "cn_edge_high": 0.5,
            "cn_start": 0.0,
            "cn_stop": 1.0,
            "cn_strength": 0.9,
            "cn_upscale": "None",
            # pose reference enables PyraCanny without a Cheat Code
            "pose_enable": bool(use_pyracanny),
            # face swap
            "faceswap_enable": bool(use_faceswap and face_image is not None),
            "faceswap_source": face_image,
            "faceswap_det_size": 640,
            "faceswap_target_index": 0,
            "faceswap_source_index": 0,
            "faceswap_paste_back": True,
            "enhance_enable": False,
        }

        progress(0.10, desc="generating")
        task_id = worker.add_task(gen_data)
        results = _wait(task_id)

        if not results:
            raise gr.Error("generation produced no image - check the server log")

        out = results[0]
        _prune_outputs()
        progress(1.0, desc="done")
        return str(out)

    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"generation failed: {exc}") from exc
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 3. enhance / upscale
# ---------------------------------------------------------------------------
def upscale_image(image, upscaler, max_edge, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image to upscale.")
    if not upscaler or upscaler == "None":
        raise gr.Error("Pick an upscaler.")

    progress(0.1, desc=f"loading {upscaler}")
    try:
        model = enhance.load_upscaler(upscaler)
        if model is None:
            raise gr.Error(
                f"could not load {upscaler} - it may still be downloading, "
                "try again in a moment"
            )

        src = image if isinstance(image, Image.Image) else Image.fromarray(image)
        progress(0.4, desc="upscaling")
        out = enhance.upscale(src, model=model)
        if out is None:
            raise gr.Error("upscaling failed - check the server log")

        out = enhance.downscale_to(out, int(max_edge))
        progress(1.0, desc="done")
        return out
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"upscale failed: {exc}") from exc


# ---------------------------------------------------------------------------
# the UI
# ---------------------------------------------------------------------------
def build_minimal_ui():
    checkpoints = _checkpoint_list()
    upscalers = _upscaler_list()

    with gr.Blocks(title="RuinedFooocus", analytics_enabled=False) as demo:
        gr.Markdown("# RuinedFooocus\nPrompt, pose + face references, upscale.")

        # ---------------------------------------------------------- generate
        with gr.Row():
            with gr.Column(scale=5):
                result = gr.Image(label="Result", type="filepath", height=560)

            with gr.Column(scale=4):
                prompt = gr.Textbox(
                    label="Prompt", lines=4,
                    placeholder="a woman in a red coat, city street, golden hour",
                )
                negative = gr.Textbox(
                    label="Negative prompt", lines=2, value=DEFAULT_NEGATIVE,
                )

                with gr.Row():
                    resolution = gr.Dropdown(
                        label="Resolution", choices=RESOLUTIONS,
                        value=DEFAULT_RESOLUTION,
                    )
                    checkpoint = gr.Dropdown(
                        label="Model", choices=checkpoints, value=checkpoints[0],
                    )

                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=10, maximum=60,
                                      value=30, step=1)
                    cfg = gr.Slider(label="CFG", minimum=1.0, maximum=12.0,
                                    value=5.0, step=0.1)

                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

                gr.Markdown(
                    "### References (optional)\n"
                    "Add a **pose** photo to follow its body and clothing "
                    "outline (PyraCanny), and a **face** photo to take the "
                    "identity from. Both together produce the final image."
                )
                with gr.Row():
                    pose_ref = gr.Image(
                        label="Pose reference (body / clothing)", type="pil", height=200,
                    )
                    face_ref = gr.Image(
                        label="Face reference (identity)", type="pil", height=200,
                    )

                with gr.Row():
                    use_pyracanny = gr.Checkbox(
                        label="PyraCanny on the pose reference", value=True,
                    )
                    use_faceswap = gr.Checkbox(
                        label="Swap the face reference in", value=True,
                    )

                run = gr.Button("Generate", variant="primary", size="lg")

        # ---------------------------------------------------------- upscale
        with gr.Accordion(label="Enhance / Upscale", open=False):
            with gr.Row():
                up_in = gr.Image(label="Image to upscale", type="pil", height=320)
                up_out = gr.Image(label="Upscaled", type="filepath", height=320)

            with gr.Row():
                upscaler = gr.Dropdown(
                    label="Upscaler",
                    choices=upscalers,
                    value="4x-UltraSharp.pth" if "4x-UltraSharp.pth" in upscalers
                    else (upscalers[0] if upscalers else None),
                    info="Downloads on first use",
                )
                max_edge = gr.Dropdown(
                    label="Final long edge",
                    choices=[1024, 1536, 2048, 2560, 3072],
                    value=2048,
                )

            up_btn = gr.Button("Upscale")

        run.click(
            generate,
            inputs=[prompt, negative, resolution, checkpoint, steps, cfg, seed,
                    pose_ref, face_ref, use_pyracanny, use_faceswap],
            outputs=[result],
        )
        up_btn.click(
            upscale_image,
            inputs=[up_in, upscaler, max_edge],
            outputs=[up_out],
        )

    return demo
