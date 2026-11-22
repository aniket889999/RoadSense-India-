"""Checkpoint trust, offline loading, and training-only adaptation boundaries."""
import hashlib
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.anpr.ocr import EasyOcrPlateRecognizer, RapidOcrPlateRecognizer, RAPIDOCR_REQUIRED_ONNX_FILES
from src.anpr.runtime_contracts import RuntimeUnavailable
from src.anpr.text_regions import plate_text_regions


def assets(tmp_path):
    root = tmp_path / "repo"
    base = root / "models/easyocr"
    base.mkdir(parents=True)
    (base / "english_g2.pth").write_bytes(b"official-test-fixture")
    adapted = root / "models/adapted.pth"
    adapted.write_bytes(b"adapted-test-fixture")
    return root, adapted, hashlib.sha256(adapted.read_bytes()).hexdigest()


def test_adapted_checkpoint_is_pinned_offline_and_restored_strictly_on_cpu(tmp_path, monkeypatch):
    root, path, digest = assets(tmp_path)
    calls = {}
    model = SimpleNamespace(load_state_dict=lambda state, strict: calls.update(state=state, strict=strict), eval=lambda: None)

    def reader(*args, **kwargs):
        calls["reader"] = kwargs
        return SimpleNamespace(recognizer=model)

    def load(path, **kwargs):
        calls["load"] = kwargs
        return {"module.layer.weight": "tensor-fixture"}

    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=reader))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))
    adapter = EasyOcrPlateRecognizer(model_dir="models/easyocr", device="cpu", repo_root=root,
                                    recognizer_weights="models/adapted.pth", recognizer_sha256=digest)
    assert calls["reader"]["download_enabled"] is False
    assert calls["reader"]["quantize"] is False
    assert calls["load"] == {"map_location": "cpu", "weights_only": True}
    assert calls["state"] == {"layer.weight": "tensor-fixture"}
    assert calls["strict"] is True
    assert adapter.recognizer_checkpoint_sha256 == digest
    assert path in adapter.weight_files


def test_wrong_adapted_hash_is_rejected_before_model_loading(tmp_path):
    root, _, _ = assets(tmp_path)
    with pytest.raises(RuntimeUnavailable, match="SHA-256 mismatch"):
        EasyOcrPlateRecognizer(model_dir="models/easyocr", device="cpu", repo_root=root,
                               recognizer_weights="models/adapted.pth", recognizer_sha256="0" * 64)


@pytest.mark.parametrize("tampered", RAPIDOCR_REQUIRED_ONNX_FILES)
def test_each_rapidocr_asset_is_hash_checked_before_engine_construction(tmp_path, monkeypatch, tampered):
    root, _, _ = assets(tmp_path)
    directory = root / "models/rapidocr"
    directory.mkdir()
    for filename in RAPIDOCR_REQUIRED_ONNX_FILES:
        (directory / filename).write_bytes(b"trusted-fixture")
    monkeypatch.setattr("src.anpr.ocr.RAPIDOCR_EXPECTED_SHA256", {
        filename: hashlib.sha256(b"trusted-fixture").hexdigest()
        for filename in RAPIDOCR_REQUIRED_ONNX_FILES
    })
    (directory / tampered).write_bytes(b"tampered")
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=lambda **k: pytest.fail("Loaded a tampered model")))
    with pytest.raises(RuntimeUnavailable, match="SHA-256 mismatch"):
        RapidOcrPlateRecognizer(model_dir="models/rapidocr", repo_root=root)


def test_rapidocr_launcher_overrides_stale_easyocr_settings_without_changing_parent():
    from scripts.serve_anpr_local import build_environment
    inherited = {"PATH": "unchanged", "ROADSENSE_ANPR_OCR_ENGINE": "easyocr",
                 "ROADSENSE_ANPR_OCR_RECOGNIZER_WEIGHTS": "old.pth"}
    actual = build_environment("rapidocr", "mps", inherited)
    assert actual["PATH"] == "unchanged"
    assert actual["ROADSENSE_ANPR_OCR_ENGINE"] == "rapidocr"
    assert actual["ROADSENSE_ANPR_OCR_MODEL_DIR"] == "models/rapidocr"
    assert actual["ROADSENSE_ANPR_OCR_CROP_VIEW"] == "resized_bgr"
    assert "ROADSENSE_ANPR_OCR_RECOGNIZER_WEIGHTS" not in actual
    assert inherited["ROADSENSE_ANPR_OCR_RECOGNIZER_WEIGHTS"] == "old.pth"


def test_easyocr_launcher_preserves_the_adapted_model_option():
    from scripts.serve_anpr_local import build_environment
    actual = build_environment("easyocr", "cpu", {})
    assert actual["ROADSENSE_ANPR_OCR_ENGINE"] == "easyocr"
    assert actual["ROADSENSE_ANPR_OCR_MODEL_DIR"] == "models/easyocr"
    assert actual["ROADSENSE_ANPR_OCR_RECOGNIZER_SHA256"] == "c72eab9cc00d2aff5a2eb006be5251d346d3be02de225db92aacbf08ad615fd3"


@pytest.mark.parametrize("path,digest", [("models/adapted.pth", ""), ("", "0" * 64)])
def test_adapted_weight_path_and_hash_are_required_together(tmp_path, path, digest):
    root, _, _ = assets(tmp_path)
    with pytest.raises(RuntimeUnavailable, match="require a model path"):
        EasyOcrPlateRecognizer(model_dir="models/easyocr", device="cpu", repo_root=root,
                               recognizer_weights=path, recognizer_sha256=digest)


def test_training_crop_collection_excludes_validation_test_and_two_line_labels(tmp_path, monkeypatch):
    from scripts import finetune_anpr_ocr as trainer
    (tmp_path / "plates.csv").write_text("image,plate_index,layout\none.jpg,0,single_line\ntwo.jpg,0,two_line\nthree.jpg,0,single_line\nfour.jpg,0,single_line\n")
    records = [SimpleNamespace(image_id="witvix-" + name, split=split, plates=[object()])
               for name, split in (("one", "train"), ("two", "train"), ("three", "val"), ("four", "test"))]
    monkeypatch.setattr(trainer, "load_manifest", lambda *a, **k: records)
    samples = trainer.collect_samples(tmp_path, tmp_path)
    assert samples == [(records[0], records[0].plates[0])]


def test_adaptation_default_is_dry_run_without_model_loading_or_output_creation(tmp_path, monkeypatch, capsys):
    from scripts import finetune_anpr_ocr as trainer
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(trainer, "validate_prepared_dataset", lambda _: {"dataset_fingerprint": "a" * 64})
    monkeypatch.setattr(trainer, "collect_samples", lambda *a: [object()])
    monkeypatch.setattr(trainer, "EasyOcrPlateRecognizer", lambda **k: pytest.fail("Dry run loaded OCR"))
    monkeypatch.setattr(sys, "argv", ["finetune", "--dataset", str(tmp_path), "--source", str(tmp_path), "--output", str(tmp_path / "outputs/dry")])
    trainer.main()
    assert '"mode": "dry_run"' in capsys.readouterr().out
    assert not (tmp_path / "outputs/dry").exists()


@pytest.mark.parametrize("background,ink", [(255, 0), (0, 255)])
def test_diagnostic_regions_find_two_visible_lines_without_mutating_pixels(background, ink):
    image = np.full((120, 220, 3), background, np.uint8)
    cv2.putText(image, "MH12", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (ink,) * 3, 2)
    cv2.putText(image, "AB1234", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (ink,) * 3, 2)
    original = image.copy()
    rows = plate_text_regions(image)
    assert len(rows) == 2
    assert rows[0][1] < rows[1][1]
    assert np.array_equal(image, original)
    assert plate_text_regions(np.zeros((30, 100, 3), np.uint8)) == []
