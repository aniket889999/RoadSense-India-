"""Launch the local ANPR API with RapidOCR review suggestions or adapted EasyOCR."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "ROADSENSE_ANPR_PLATE_WEIGHTS": "models/anpr/anpr_witvix_detector_mps_v1_best.pt",
    "ROADSENSE_ANPR_PLATE_SHA256": "37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db",
    "ROADSENSE_ANPR_OCR_MODEL_DIR": "models/easyocr",
    "ROADSENSE_ANPR_OCR_CROP_VIEW": "original_bgr",
    "ROADSENSE_ANPR_OCR_RECOGNIZER_WEIGHTS": "models/anpr_ocr/english_g2_witvix_v1.pth",
    "ROADSENSE_ANPR_OCR_RECOGNIZER_SHA256": "c72eab9cc00d2aff5a2eb006be5251d346d3be02de225db92aacbf08ad615fd3",
    "ROADSENSE_ANPR_DEVICE": "mps",
}


def build_environment(engine: str, device: str, inherited=None):
    if engine not in ("rapidocr", "easyocr"):
        raise ValueError("Select rapidocr or easyocr explicitly.")
    env = {**(os.environ if inherited is None else inherited), **DEFAULTS,
           "ROADSENSE_ANPR_DEVICE": device, "ROADSENSE_ANPR_OCR_ENGINE": engine}
    if engine == "rapidocr":
        env["ROADSENSE_ANPR_OCR_MODEL_DIR"] = "models/rapidocr"
        env["ROADSENSE_ANPR_OCR_CROP_VIEW"] = "resized_bgr"
        env.pop("ROADSENSE_ANPR_OCR_RECOGNIZER_WEIGHTS", None)
        env.pop("ROADSENSE_ANPR_OCR_RECOGNIZER_SHA256", None)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--engine", choices=("rapidocr", "easyocr"), default="rapidocr",
                        help="RapidOCR uses CPU OCR and keeps every read for human review; device selects the plate detector.")
    args = parser.parse_args()
    env = build_environment(args.engine, args.device)
    os.chdir(ROOT)
    os.execve(str(ROOT / ".venv/bin/python"), [str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "services.api.app.main:app", "--host", "127.0.0.1", "--port", str(args.port)], env)
