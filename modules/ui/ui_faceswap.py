"""Face Swap UI.

Rendered as a section *inside* the PowerUp tab, not as a top-level tab, so it
sits directly under the Cheat Code dropdown where it is impossible to miss.
"""

import gradio as gr

import modules.faceswap as faceswap
from shared import add_ctrl, path_manager, translate

t = translate


def add_faceswap_section():
    """Build the Face Swap + Enhance controls. Call inside an existing tab."""
    with gr.Accordion(label=t("Face Swap"), open=True):
        gr.Markdown(
            value=(
                "Replaces the face in the generated image with the face from the "
                "**Source face image** below. This is a real identity swap "
                "(insightface inswapper), applied after generation, so it works "
                "on batches too.\n\n"
                "The first run downloads about 850 MB into `models/faceswap/`."
            )
        )

        with gr.Row():
            source_image = gr.Image(
                label=t("Source face image"),
                type="pil",
            )
            add_ctrl("faceswap_source", source_image)

            with gr.Column():
                enable = gr.Checkbox(
                    label=t("Enable face swap"),
                    value=False,
                )
                add_ctrl("faceswap_enable", enable)

                det_size = gr.Dropdown(
                    label=t("Detector size"),
                    choices=[256, 320, 512, 640, 768, 1024],
                    value=640,
                    info=t("Larger finds smaller faces but is slower"),
                )
                add_ctrl("faceswap_det_size", det_size)

                target_index = gr.Number(
                    label=t("Target face #"),
                    value=0,
                    precision=0,
                    info=t("Which face in the generated image, largest first"),
                )
                add_ctrl("faceswap_target_index", target_index)

                source_index = gr.Number(
                    label=t("Source face #"),
                    value=0,
                    precision=0,
                    info=t("Which face in the source image, largest first"),
                )
                add_ctrl("faceswap_source_index", source_index)

                paste_back = gr.Checkbox(
                    label=t("Blend into the image"),
                    value=True,
                    info=t("Off pastes the raw 128px swapped crop instead"),
                )
                add_ctrl("faceswap_paste_back", paste_back)

    with gr.Accordion(label=t("Enhance / Upscale"), open=False):
        gr.Markdown(
            value=(
                "Runs after generation. The upscaler enlarges the image, then "
                "the face swap is applied **last** — a 128px swap stretched by "
                "an upscaler goes soft, so doing it at the end keeps the "
                "identity sharp."
            )
        )

        enhance_enable = gr.Checkbox(
            label=t("Enable enhance / upscale"),
            value=False,
        )
        add_ctrl("enhance_enable", enhance_enable)

        enhance_upscaler = gr.Dropdown(
            label=t("Upscaler"),
            choices=["None"] + path_manager.upscaler_filenames,
            value="4x-UltraSharp.pth",
            info=t("First use downloads the model"),
        )
        add_ctrl("enhance_upscaler", enhance_upscaler)

        enhance_max_edge = gr.Dropdown(
            label=t("Final long edge"),
            choices=[1024, 1536, 2048, 2560, 3072],
            value=2048,
            info=t("The 4x result is scaled down to this"),
        )
        add_ctrl("enhance_max_edge", enhance_max_edge)

    with gr.Accordion(label=t("Realistic preset"), open=False):
        gr.Markdown(
            value=(
                "Applies photographic styles, CFG 5, 40 steps and a 3:4 "
                "resolution — the settings that read as a real photo rather "
                "than an illustration."
            )
        )
        preset_btn = gr.Button(value=t("Apply realistic preset"))
        preset_note = gr.Textbox(label="", value="", lines=2, interactive=False)

        def apply_realistic():
            from modules.enhance import REALISTIC_PRESET

            return [{k: gr.update(value=v) for k, v in REALISTIC_PRESET.items()},
                    "Applied."]

        from shared import get_ctrl

        targets = []
        for key in ("style_selection", "cfg", "custom_steps",
                    "sampler_name", "scheduler", "negative"):
            try:
                ctrl = get_ctrl(key)
            except (ValueError, KeyError):
                ctrl = None
            if ctrl is not None:
                targets.append(ctrl)

        if targets:
            preset_btn.click(
                api_visibility="undocumented",
                fn=apply_realistic,
                inputs=[],
                outputs=targets + [preset_note],
            )

    with gr.Accordion(label=t("Face swap status"), open=False):
        status_box = gr.Textbox(label="", value="", lines=4, interactive=False)
        check_btn = gr.Button(value=t("Check models"))

        check_btn.click(
            api_visibility="undocumented",
            fn=lambda: faceswap.status(),
            inputs=[],
            outputs=[status_box],
        )

    return enable


def add_faceswap_tab():
    """Kept for compatibility: a tab that contains the section."""
    with gr.Tab(label=t("Face Swap")):
        return add_faceswap_section()
