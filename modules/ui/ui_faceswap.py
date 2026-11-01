"""Face Swap tab.

A standalone tab rather than a ControlNet "Cheat Code" entry, because a swap is
a post-process applied to the finished image, not a conditioning signal fed into
the sampler. Keeping it separate also means it can be turned on without touching
the ControlNet slots.
"""

import gradio as gr

import modules.faceswap as faceswap
from shared import add_ctrl, path_manager, translate

t = translate


def add_faceswap_tab():
    with gr.Tab(label=t("Face Swap")):
        gr.Markdown(
            value=(
                "Replaces the face in the generated image with the face from the "
                "**Source image**. This is a real identity swap (insightface "
                "inswapper), applied after generation, so it also works on "
                "batches and on inpainted results.\n\n"
                "Set **Source image** and tick **Enable face swap**. The first "
                "run downloads about 850 MB into `models/faceswap/`."
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

        with gr.Accordion(label=t("Status"), open=False):
            status_box = gr.Textbox(
                label="",
                value="",
                lines=4,
                interactive=False,
            )

            check_btn = gr.Button(value=t("Check models"))

            def check_status():
                return faceswap.status()

            check_btn.click(
                api_visibility="undocumented",
                fn=check_status,
                inputs=[],
                outputs=[status_box],
            )

        # Escape hatch for hosts that proxy the UI under a path (Colab does).
        # Gradio builds this link from its own live server address.
        with gr.Row():
            open_link = gr.HTML(value="")

            def show_link(request: gr.Request):
                url = str(request.base_url).rstrip("/")
                return (
                    f'<a href="{url}" target="_blank" '
                    f'style="font-size:13px">Open the UI in its own tab</a>'
                )

            enable.change(
                api_visibility="undocumented",
                fn=show_link,
                inputs=[],
                outputs=[open_link],
            )

    return enable
