"""Minimal RuinedFooocus entry point.

Replaces the original full UI (tabs, LoRAs, styles, presets, cheat codes,
inpainting, image browser, chat, settings, model manager) with three things:

  1. prompt -> image
  2. pose reference (PyraCanny) + face reference (swap) -> image
  3. an image -> upscale

The backend is untouched: modules/async_worker.py, modules/image_pipeline.py,
modules/pipelines.py and every model pipeline still do the real work.
"""

import time
import traceback

import gradio as gr

import shared
import modules.html
import version
from argparser import args
from modules.ui.ui_minimal import build_minimal_ui


def launch_app(args):
    demo = build_minimal_ui()

    shared.gradio_root = demo
    demo.queue(api_open=True)

    favicon_path = "logo.ico"
    try:
        shared.server_app, shared.local_url, shared.share_url = demo.launch(
            inbrowser=not args.nobrowser,
            server_name=args.listen,
            server_port=args.port,
            share=args.share,
            auth=(
                args.auth.split("/", 1)
                if isinstance(args.auth, str) and "/" in args.auth
                else None
            ),
            favicon_path=favicon_path,
            css=modules.html.css,
            js=modules.html.scripts,
            allowed_paths=[str(p) for p in
                           [shared.path_manager.model_paths["temp_outputs_path"],
                            shared.path_manager.model_paths["upscaler_path"],
                            shared.path_manager.model_paths["faceswap_path"],
                            "html"]],
            enable_monitoring=False,
            prevent_thread_lock=True,
        )
    except Exception as exc:
        print(f"ERROR: could not launch the UI: {exc}")
        traceback.print_exc()
        raise


print(f"Starting RuinedFooocus {version.version} (minimal UI)")
launch_app(args)

# Keep the process alive; launch() is non-blocking.
while True:
    time.sleep(100)
