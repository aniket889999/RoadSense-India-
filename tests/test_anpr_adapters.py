"""Focused tests for ANPR detector, preprocessing, EasyOCR and local runtime adapters."""

from __future__ import annotations

import hashlib
from io import BytesIO
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.anpr.contracts import PlateBox
from src.anpr.detector import (
    ENV_ANPR_DEVICE,
    ENV_PLATE_SHA256,
    ENV_PLATE_WEIGHTS,
    UltralyticsPlateDetector,
    _validate_single_plate_class,
)
from src.anpr.local_runtime import build_local_runtime
from src.anpr.pipeline import AnprImageService
from src.anpr.ocr import (
    ENV_OCR_ENGINE,
    ENV_OCR_MODEL_DIR,
    EasyOcrPlateRecognizer,
    RAPIDOCR_REQUIRED_ONNX_FILES,
    RapidOcrPlateRecognizer,
    order_and_merge_ocr_detections,
)
from src.anpr.preprocessing import (
    MAX_OCR_HEIGHT_PX,
    MIN_OCR_HEIGHT_PX,
    extract_plate_crop,
    preprocess_plate_crop,
)
from src.anpr.runtime_contracts import RuntimeUnavailable


class _FakeArray:
    def __init__(self, rows):
        self._rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._rows)


def _setup_fake_repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    """Create an isolated repository root with models/ for confinement tests."""
    repo_root = tmp_path / "repo"
    models_dir = repo_root / "models"
    anpr_dir = models_dir / "anpr"
    ocr_dir = models_dir / "easyocr"
    anpr_dir.mkdir(parents=True)
    ocr_dir.mkdir(parents=True)

    weights_file = anpr_dir / "plate_detector.pt"
    weights_bytes = b"verified-single-class-plate-detector"
    weights_file.write_bytes(weights_bytes)
    weights_sha256 = hashlib.sha256(weights_bytes).hexdigest()

    (ocr_dir / "craft_mlt_25k.pth").write_bytes(b"local-craft-weights")
    (ocr_dir / "english_g2.pth").write_bytes(b"local-english-weights")

    return (
        repo_root,
        "models/anpr/plate_detector.pt",
        weights_sha256,
        "models/easyocr",
    )


def test_importing_adapter_modules_does_not_load_heavy_ml_packages():
    """Importing the ANPR adapter modules must not eagerly import torch, ultralytics or easyocr."""
    code = (
        "import sys; "
        "import src.anpr.detector, src.anpr.preprocessing, src.anpr.ocr, src.anpr.local_runtime; "
        "loaded = [m for m in ('torch', 'ultralytics', 'easyocr') if m in sys.modules]; "
        "assert not loaded, f'Unexpected eager imports: {loaded}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_detector_requires_configured_weights_and_sha256(monkeypatch, tmp_path):
    repo_root, rel_weights, sha256, _ = _setup_fake_repo(tmp_path)
    monkeypatch.delenv(ENV_PLATE_WEIGHTS, raising=False)
    monkeypatch.delenv(ENV_PLATE_SHA256, raising=False)

    with pytest.raises(RuntimeUnavailable, match=ENV_PLATE_SHA256):
        UltralyticsPlateDetector(repo_root=repo_root)

    with pytest.raises(RuntimeUnavailable, match=ENV_PLATE_WEIGHTS):
        UltralyticsPlateDetector(expected_sha256=sha256, repo_root=repo_root)

    with pytest.raises(RuntimeUnavailable, match="SHA-256 mismatch"):
        UltralyticsPlateDetector(
            weights_path=rel_weights,
            expected_sha256="0" * 64,
            repo_root=repo_root,
        )


def test_detector_rejects_escape_and_symlinks_before_loading_yolo(monkeypatch, tmp_path):
    repo_root, _, sha256, _ = _setup_fake_repo(tmp_path)
    outside_file = tmp_path / "outside.pt"
    outside_file.write_bytes(b"outside")
    outside_sha = hashlib.sha256(b"outside").hexdigest()

    symlink_path = repo_root / "models" / "anpr" / "symlink_plate.pt"
    symlink_path.symlink_to(outside_file)

    yolo_calls = []

    class FakeYOLO:
        def __init__(self, path):
            yolo_calls.append(path)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    for bad_path in [
        str(outside_file),
        "../outside.pt",
        "models/../../outside.pt",
        "README.md",
        "models/anpr/symlink_plate.pt",
    ]:
        with pytest.raises(RuntimeUnavailable):
            UltralyticsPlateDetector(
                weights_path=bad_path,
                expected_sha256=outside_sha if "symlink" in bad_path else sha256,
                repo_root=repo_root,
            )

    assert yolo_calls == []


@pytest.mark.parametrize(
    "invalid_names",
    [
        {0: "pothole"},
        {0: "car", 1: "truck", 2: "bus"},
        {0: "license_plate", 1: "vehicle"},
        ["pothole"],
        {},
    ],
)
def test_detector_rejects_pothole_or_multi_class_vehicle_checkpoints(
    monkeypatch, tmp_path, invalid_names
):
    repo_root, rel_weights, sha256, _ = _setup_fake_repo(tmp_path)

    class FakeYOLO:
        def __init__(self, path):
            self.path = path
            self.names = invalid_names

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(RuntimeUnavailable, match="license_plate"):
        UltralyticsPlateDetector(
            weights_path=rel_weights,
            expected_sha256=sha256,
            repo_root=repo_root,
        )


@pytest.mark.parametrize("invalid_names", [
    {0.5: "license_plate"},
    {False: "license_plate"},
    {0: "license_plate", "0": "license_plate"},
])
def test_class_ids_cannot_be_coerced_or_collapsed_into_a_valid_plate_class(invalid_names):
    with pytest.raises(RuntimeUnavailable):
        _validate_single_plate_class(types.SimpleNamespace(names=invalid_names))


def test_detector_runs_single_class_inference_and_clamps_boxes(monkeypatch, tmp_path):
    repo_root, rel_weights, sha256, _ = _setup_fake_repo(tmp_path)
    predict_calls = []

    class FakeYOLO:
        def __init__(self, path):
            self.path = path
            self.names = {0: "license_plate"}

        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            boxes = types.SimpleNamespace(
                xyxy=_FakeArray(
                    [
                        [120.0, 80.0, 260.0, 130.0],
                        [-5.0, 10.0, 95.0, 45.0],
                        [10.0, 10.0, 10.0, 30.0],  # degenerate zero-width box ignored
                        [15.0, 15.0, 50.0, 40.0],  # non-zero class ignored
                    ]
                ),
                conf=_FakeArray([0.91, 0.84, 0.95, 0.99]),
                cls=_FakeArray([0, 0, 0, 1]),
            )
            return [types.SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.delenv(ENV_ANPR_DEVICE, raising=False)

    detector = UltralyticsPlateDetector(
        weights_path=rel_weights,
        expected_sha256=sha256,
        repo_root=repo_root,
    )
    assert detector.device == "cpu"

    image = np.zeros((200, 300, 3), dtype=np.uint8)
    detections = detector.detect(image)

    assert len(predict_calls) == 1
    assert predict_calls[0]["device"] == "cpu"
    assert predict_calls[0]["classes"] == [0]
    assert len(detections) == 2
    assert detections[0].bbox == PlateBox(x1=0.0, y1=10.0, x2=95.0, y2=45.0)
    assert detections[0].confidence == pytest.approx(0.84)
    assert detections[1].bbox == PlateBox(x1=120.0, y1=80.0, x2=260.0, y2=130.0)
    assert detections[1].confidence == pytest.approx(0.91)


def test_preprocessing_preserves_original_crop_and_bounds_dimensions():
    source = np.random.default_rng(42).integers(0, 256, size=(120, 240, 3), dtype=np.uint8)
    source_snapshot = source.copy()
    bbox = PlateBox(x1=30, y1=40, x2=150, y2=60)

    crop = extract_plate_crop(source, bbox, padding_ratio=0.0)
    assert np.array_equal(source, source_snapshot)
    assert crop.shape == (20, 120, 3)

    crop_snapshot = crop.copy()
    prep = preprocess_plate_crop(crop)

    assert np.array_equal(crop, crop_snapshot)
    assert np.array_equal(prep.original_bgr, crop_snapshot)
    assert prep.original_bgr is not crop
    assert MIN_OCR_HEIGHT_PX <= prep.resized_bgr.shape[0] <= MAX_OCR_HEIGHT_PX
    assert prep.grayscale.shape == prep.resized_bgr.shape[:2]
    assert prep.enhanced_grayscale.shape == prep.resized_bgr.shape[:2]
    assert prep.enhanced_grayscale.dtype == np.uint8


def test_ocr_orders_two_line_and_single_line_boxes_and_takes_conservative_score():
    # Two-line plate passed out of order: bottom-right, top-right, bottom-left, top-left
    detections = [
        ([[60, 42], [130, 42], [130, 72], [60, 72]], "1234", 0.93),
        ([[55, 5], [110, 5], [110, 33], [55, 33]], "12", 0.88),
        ([[8, 40], [54, 40], [54, 70], [8, 70]], "AB", 0.79),
        ([[6, 4], [50, 4], [50, 32], [6, 32]], "MH", 0.95),
    ]
    read = order_and_merge_ocr_detections(detections)
    assert read.raw_text == "MH 12\nAB 1234"
    assert read.confidence == pytest.approx(0.79)


def test_ocr_never_silently_truncates_original_text():
    detections = [
        ([[0, 0], [300, 0], [300, 30], [0, 30]], "MH12AB1234" + " " * 247, 0.99),
    ]
    with pytest.raises(RuntimeUnavailable, match="refusing to truncate"):
        order_and_merge_ocr_detections(detections)


def test_ocr_requires_local_assets_and_disables_downloads(monkeypatch, tmp_path):
    repo_root, _, _, rel_ocr_dir = _setup_fake_repo(tmp_path)
    reader_calls = []

    class FakeReader:
        def __init__(self, lang_list, **kwargs):
            reader_calls.append((lang_list, kwargs))

        def readtext(self, image, detail=1, paragraph=False):
            assert image.ndim == 3 and image.shape[2] == 3
            return [
                ([[50, 8], [140, 8], [140, 36], [50, 36]], "12 AB 1234", 0.86),
                ([[4, 8], [44, 8], [44, 36], [4, 36]], "MH", 0.94),
            ]

    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=FakeReader))

    recognizer = EasyOcrPlateRecognizer(
        model_dir=rel_ocr_dir,
        device="cpu",
        repo_root=repo_root,
    )
    assert len(reader_calls) == 1
    langs, kwargs = reader_calls[0]
    assert langs == ["en"]
    assert kwargs["download_enabled"] is False
    assert kwargs["gpu"] is False
    assert kwargs["model_storage_directory"] == str(repo_root / rel_ocr_dir)

    plate_crop = np.full((28, 120, 3), 180, dtype=np.uint8)
    result = recognizer.recognize(plate_crop)
    assert recognizer.crop_view == "original_bgr"
    assert result.raw_text == "MH 12 AB 1234"
    assert result.confidence == pytest.approx(0.86)

    with pytest.raises(ValueError, match="crop_view"):
        EasyOcrPlateRecognizer(
            model_dir=rel_ocr_dir,
            device="cpu",
            repo_root=repo_root,
            crop_view="invalid_view",
        )


def test_ocr_rejects_empty_or_symlinked_model_dir(monkeypatch, tmp_path):
    repo_root, _, _, _ = _setup_fake_repo(tmp_path)
    empty_dir = repo_root / "models" / "empty_ocr"
    empty_dir.mkdir(parents=True)

    with pytest.raises(RuntimeUnavailable, match="No local EasyOCR weight files"):
        EasyOcrPlateRecognizer(model_dir="models/empty_ocr", repo_root=repo_root)

    outside_weight = tmp_path / "external.pth"
    outside_weight.write_bytes(b"external")
    (empty_dir / "linked.pth").symlink_to(outside_weight)

    with pytest.raises(RuntimeUnavailable, match="symlink"):
        EasyOcrPlateRecognizer(model_dir="models/empty_ocr", repo_root=repo_root)


def test_build_local_runtime_composes_adapters_and_fails_closed_when_unconfigured(
    monkeypatch, tmp_path
):
    repo_root, rel_weights, sha256, rel_ocr_dir = _setup_fake_repo(tmp_path)

    for env_key in (ENV_PLATE_WEIGHTS, ENV_PLATE_SHA256, ENV_OCR_MODEL_DIR, ENV_ANPR_DEVICE):
        monkeypatch.delenv(env_key, raising=False)

    with pytest.raises(RuntimeUnavailable):
        build_local_runtime(repo_root=repo_root)

    class FakeYOLO:
        def __init__(self, path):
            self.names = {0: "license_plate"}

        def predict(self, **kwargs):
            return []

    class FakeReader:
        def __init__(self, lang_list, **kwargs):
            assert kwargs["download_enabled"] is False

        def readtext(self, image, detail=1, paragraph=False):
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=FakeReader))
    monkeypatch.setenv(ENV_PLATE_WEIGHTS, rel_weights)
    monkeypatch.setenv(ENV_PLATE_SHA256, sha256)
    monkeypatch.setenv(ENV_OCR_MODEL_DIR, rel_ocr_dir)

    runtime = build_local_runtime(repo_root=repo_root)
    frame = np.zeros((64, 160, 3), dtype=np.uint8)
    assert runtime.detector.detect(frame) == []
    assert runtime.recognizer.recognize(frame).raw_text == ""


def test_composed_adapters_integrate_with_the_image_pipeline(monkeypatch, tmp_path):
    """Run decoding, real adapters, preprocessing and validation with fake ML engines."""
    repo_root, rel_weights, sha256, rel_ocr_dir = _setup_fake_repo(tmp_path)

    class FakeYOLO:
        def __init__(self, path):
            self.names = {0: "license_plate"}

        def predict(self, **kwargs):
            assert kwargs["source"].shape == (80, 160, 3)
            boxes = types.SimpleNamespace(
                xyxy=_FakeArray([[10, 15, 145, 60]]),
                conf=_FakeArray([0.95]),
                cls=_FakeArray([0]),
            )
            return [types.SimpleNamespace(boxes=boxes)]

    class FakeReader:
        def __init__(self, languages, **kwargs):
            assert kwargs["download_enabled"] is False

        def readtext(self, image, **kwargs):
            assert image.ndim == 3 and image.shape[2] == 3
            return [([[2, 2], [120, 2], [120, 35], [2, 35]], "MH 12 AB 1234", 0.91)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=FakeReader))

    service = AnprImageService(lambda: build_local_runtime(
        weights_path=rel_weights,
        expected_sha256=sha256,
        ocr_model_dir=rel_ocr_dir,
        device="cpu",
        repo_root=repo_root,
    ))
    encoded = BytesIO()
    Image.new("RGB", (160, 80), "white").save(encoded, "PNG")
    result = service.recognize_bytes(encoded.getvalue(), "image/png", "adapter-integration")

    assert result.status == "COMPLETE"
    assert result.image_width == 160
    assert result.image_height == 80
    assert result.plates[0].bbox == PlateBox(x1=10, y1=15, x2=145, y2=60)
    assert result.plates[0].raw_text == "MH 12 AB 1234"
    assert result.plates[0].normalized_text == "MH12AB1234"
    assert result.plates[0].format_status == "MATCHED"


def test_rapidocr_recognizer_orders_two_line_boxes_and_keeps_review_only_confidence_none(
    monkeypatch, tmp_path
):
    repo_root, rel_weights, sha256, _ = _setup_fake_repo(tmp_path)
    rapid_dir = repo_root / "models" / "rapidocr"
    rapid_dir.mkdir(parents=True)
    for fname in RAPIDOCR_REQUIRED_ONNX_FILES:
        (rapid_dir / fname).write_bytes(b"fake-onnx-model-bytes")
    monkeypatch.setattr("src.anpr.ocr.RAPIDOCR_EXPECTED_SHA256", {
        fname: hashlib.sha256(b"fake-onnx-model-bytes").hexdigest()
        for fname in RAPIDOCR_REQUIRED_ONNX_FILES
    })

    init_calls = []

    class FakeRapidOCR:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def __call__(self, image, **kwargs):
            return (
                [
                    ([[60, 42], [130, 42], [130, 72], [60, 72]], "1234", 0.96),
                    ([[6, 4], [50, 4], [50, 32], [6, 32]], "MH12", 0.98),
                    ([[8, 40], [54, 40], [54, 70], [8, 70]], "AB", 0.94),
                ],
                [0.01, 0.01, 0.01],
            )

    monkeypatch.setitem(
        sys.modules,
        "rapidocr_onnxruntime",
        types.SimpleNamespace(RapidOCR=FakeRapidOCR),
    )

    recognizer = RapidOcrPlateRecognizer(
        model_dir="models/rapidocr",
        repo_root=repo_root,
        review_only=True,
    )
    assert len(init_calls) == 1
    assert init_calls[0]["det_model_path"] == str(rapid_dir / "ch_PP-OCRv4_det_infer.onnx")

    crop = np.full((32, 120, 3), 200, dtype=np.uint8)
    read = recognizer.recognize(crop)
    assert read.raw_text == "MH12\nAB 1234"
    assert read.confidence is None

    # Also verify build_local_runtime selects RapidOcrPlateRecognizer when ENV_OCR_ENGINE=rapidocr
    class FakeYOLO:
        def __init__(self, path):
            self.names = {0: "license_plate"}

        def predict(self, **kwargs):
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setenv(ENV_PLATE_WEIGHTS, rel_weights)
    monkeypatch.setenv(ENV_PLATE_SHA256, sha256)
    monkeypatch.setenv(ENV_OCR_MODEL_DIR, "models/rapidocr")
    monkeypatch.setenv(ENV_OCR_ENGINE, "rapidocr")

    rt = build_local_runtime(repo_root=repo_root)
    assert isinstance(rt.recognizer, RapidOcrPlateRecognizer)
    assert rt.recognizer.recognize(crop).confidence is None
