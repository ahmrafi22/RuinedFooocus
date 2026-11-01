"""
InsightFace face swap for RuinedFooocus.

This is a real identity transfer (inswapper_128), not an IP-Adapter prompt
nudge: the target face is located, cropped, replaced with the source face's
identity and pasted back into the image.

Why a standalone module
-----------------------
It only needs onnxruntime + opencv + numpy, all of which RuinedFooocus already
installs, so it does not drag in ComfyUI nodes or a second torch stack.

Where it is applied
-------------------
modules/async_worker.py calls swap_image() on each finished image, right
before the file is written. That means it works for every pipeline (SDXL,
Flux, video frames, ...) and for every image in a batch.

Models
------
  models/faceswap/inswapper_128.onnx        the swapper (554 MB)
  models/faceswap/insightface/*.onnx        buffalo_l detector/recognition
                                            (unpacked from buffalo_l.zip)

Both are downloaded on first use. Set FACESWAP_NO_DOWNLOAD=1 to make a
missing model a hard error instead of a download.
"""

import os
import shutil
import threading
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# The two archives below are the official InsightFace model-zoo releases.
BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/"
    "model-zoo/buffalo_l.zip"
)

_lock = threading.RLock()
_analyzer = None
_analyzer_size = None
_swapper = None


# ---------------------------------------------------------------------------
# providers / device
# ---------------------------------------------------------------------------
def _providers():
    """Prefer the GPU when onnxruntime was built with it, else fall back."""
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]

    for provider in (
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
    ):
        if provider in available:
            return [provider, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _faceswap_dir():
    from shared import path_manager

    return Path(path_manager.model_paths["faceswap_path"])


# ---------------------------------------------------------------------------
# model handling
# ---------------------------------------------------------------------------
def _ensure_buffalo_l(target_dir):
    """Make sure target_dir holds buffalo_l's .onnx files.

    Returns the directory, or None if it is unavailable.
    """
    target_dir = Path(target_dir)
    if target_dir.is_dir() and any(target_dir.glob("*.onnx")):
        return target_dir

    if os.environ.get("FACESWAP_NO_DOWNLOAD"):
        print(f"ERROR: no insightface models in {target_dir} and downloads are disabled")
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "buffalo_l.zip"
    try:
        import requests
        from tqdm import tqdm

        print(f"Downloading {BUFFALO_L_URL} ...")
        response = requests.get(BUFFALO_L_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as handle, tqdm(
            desc="buffalo_l.zip", total=total, unit="iB", unit_scale=True
        ) as progress:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
                progress.update(len(chunk))

        # The archive contains a single "buffalo_l/" folder; flatten it so the
        # .onnx files sit directly in target_dir.
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                name = os.path.basename(member)
                if not name.lower().endswith(".onnx"):
                    continue
                with archive.open(member) as src, open(target_dir / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        zip_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"ERROR: could not prepare insightface models: {exc}")
        zip_path.unlink(missing_ok=True)
        return None

    return target_dir if any(target_dir.glob("*.onnx")) else None


def _inswapper_path():
    from shared import path_manager

    return path_manager.get_file_path("faceswap/inswapper_128.onnx")


def get_analyzer(det_size=640):
    """Load (and cache) the InsightFace analysis pipeline."""
    global _analyzer, _analyzer_size

    with _lock:
        if _analyzer is not None and _analyzer_size == det_size:
            return _analyzer

        from insightface.app import FaceAnalysis

        # Passing an absolute directory makes FaceAnalysis read the .onnx files
        # from there directly, instead of looking in <root>/models/<name> and
        # trying to download the archive itself.
        model_dir = _ensure_buffalo_l(_faceswap_dir() / "insightface")
        if model_dir is None:
            raise RuntimeError(
                "insightface models are missing; see models/faceswap/insightface"
            )

        app = FaceAnalysis(
            name=str(model_dir),
            providers=_providers(),
        )
        app.prepare(ctx_id=0, det_size=(det_size, det_size))

        _analyzer = app
        _analyzer_size = det_size
        return app


def get_swapper():
    """Load (and cache) the inswapper_128 model."""
    global _swapper

    with _lock:
        if _swapper is not None:
            return _swapper

        import onnxruntime as ort
        from insightface.model_zoo.inswapper import INSwapper

        model_path = _inswapper_path()
        if model_path is None or not Path(model_path).exists():
            raise RuntimeError(
                "inswapper_128.onnx is missing; expected it in models/faceswap/"
            )

        session = ort.InferenceSession(str(model_path), providers=_providers())
        _swapper = INSwapper(model_file=str(model_path), session=session)
        return _swapper


def unload():
    """Drop the cached models (frees a few hundred MB of RAM)."""
    global _analyzer, _analyzer_size, _swapper
    with _lock:
        _analyzer = None
        _analyzer_size = None
        _swapper = None


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------
def to_bgr(image):
    """PIL.Image / ndarray / path -> BGR uint8 ndarray (what InsightFace wants)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        array = array.astype(np.uint8)
    # InsightFace works in BGR.
    return np.ascontiguousarray(array[:, :, ::-1])


def to_pil(bgr):
    """BGR uint8 ndarray -> PIL.Image."""
    return Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))


def _pick_face(faces, index):
    """Choose a face, ordered largest-first so 0 is the main subject."""
    if not faces:
        return None
    ordered = sorted(
        faces,
        key=lambda face: float(
            (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        ),
        reverse=True,
    )
    if index is None:
        index = 0
    index = int(index)
    if index < 0 or index >= len(ordered):
        return ordered[0]
    return ordered[index]


# ---------------------------------------------------------------------------
# the actual swap
# ---------------------------------------------------------------------------
def swap_image(
    target,
    source,
    det_size=640,
    target_index=0,
    source_index=0,
    paste_back=True,
):
    """Swap the source face onto the target image.

    target/source: PIL.Image, ndarray or path.
    Returns a PIL.Image, or None if either image has no detectable face.
    """
    det_size = int(det_size)
    if det_size not in (128, 256, 320, 512, 640, 768, 1024):
        det_size = 640

    app = get_analyzer(det_size)
    swapper = get_swapper()

    target_bgr = to_bgr(target)
    source_bgr = to_bgr(source)

    target_face = _pick_face(app.get(target_bgr), target_index)
    if target_face is None:
        print("WARNING: face swap skipped, no face found in the generated image")
        return None

    source_face = _pick_face(app.get(source_bgr), source_index)
    if source_face is None:
        print("WARNING: face swap skipped, no face found in the source image")
        return None

    swapped = swapper.get(
        target_bgr, target_face, source_face, paste_back=bool(paste_back)
    )
    return to_pil(swapped)


def status():
    """Short human-readable state, for the UI."""
    directory = _faceswap_dir()
    swapper = directory / "inswapper_128.onnx"
    detector_dir = directory / "insightface"
    detector = detector_dir.is_dir() and any(detector_dir.glob("*.onnx"))
    lines = [
        f"swapper: {'ready' if swapper.exists() else 'will download on first use'}",
        f"detector: {'ready' if detector else 'will download on first use'}",
        f"providers: {', '.join(_providers())}",
    ]
    return "\n".join(lines)
