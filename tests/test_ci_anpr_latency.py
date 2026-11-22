"""Asset-free regression tests for the validation-only latency CLI.

Kept under the engineering task's permitted test_ci_* file boundary.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts import benchmark_anpr_latency as latency


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(latency, "ROOT", tmp_path)
    root = tmp_path / "data" / "processed" / "pilot"
    for split in ("train", "val", "test"):
        folder = root / "images" / split
        folder.mkdir(parents=True)
        Image.new("RGB", (96, 48), "white").save(folder / "sample.png")
    (root / "preparation_metadata.json").write_text(json.dumps({
        "dataset_fingerprint": "a" * 64, "split_counts": {"val": {"images": 1}},
    }))
    return root


@pytest.mark.parametrize("split", ["train", "test"])
def test_non_validation_path_rejected_before_model_or_image_read(dataset, monkeypatch, split):
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected input must not construct a runtime or read image bytes")

    monkeypatch.setattr(latency, "run_benchmark", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert latency.main([
        "--dataset", str(dataset), "--image", str(dataset / "images" / split / "sample.png"),
        "--device", "cpu", "--output", str(latency.ROOT / "outputs" / "timing.json"),
    ]) == 1
    assert not (latency.ROOT / "outputs").exists()


def test_validation_alias_to_test_is_rejected(dataset):
    alias = dataset / "images" / "val" / "sample.png"
    alias.unlink()
    alias.symlink_to(dataset / "images" / "test" / "sample.png")
    with pytest.raises(latency.BenchmarkError, match="Symlink"):
        latency.validation_images(dataset, [str(alias)])


def test_test_traversal_spelling_is_rejected(dataset):
    spelling = str(dataset / "images" / "test" / ".." / "val" / "sample.png")
    with pytest.raises(latency.BenchmarkError, match="Held-out test"):
        latency.validation_images(dataset, [spelling])


def test_only_validation_directory_is_enumerated(dataset, monkeypatch):
    original = Path.iterdir

    def guarded(path):
        assert path == dataset / "images" / "val"
        return original(path)

    monkeypatch.setattr(Path, "iterdir", guarded)
    selected, _ = latency.validation_images(dataset, [])
    assert selected == [dataset / "images" / "val" / "sample.png"]


def test_output_refuses_existing_or_outside_files(dataset):
    existing = latency.ROOT / "outputs" / "existing.json"
    existing.parent.mkdir()
    existing.write_text("preserve")
    with pytest.raises(latency.BenchmarkError, match="overwritten"):
        latency.output_path(existing)
    with pytest.raises(latency.BenchmarkError, match="outputs"):
        latency.output_path(latency.ROOT / "outside.json")
    assert existing.read_text() == "preserve"


def synthetic_runtime(on_detect=lambda: None):
    from src.anpr.contracts import PlateBox
    from src.anpr.runtime_contracts import AnprRuntime, OcrRead, PlateDetection

    def detect(image):
        on_detect()
        return [PlateDetection(PlateBox(x1=32, y1=8, x2=80, y2=32), 0.9)]

    return AnprRuntime(
        detector=SimpleNamespace(detect=detect),
        recognizer=SimpleNamespace(recognize=lambda crop: OcrRead("sensitive-reading-fixture", None, True)),
    )


def test_warmups_excluded_and_readings_never_enter_report(dataset):
    selected, _ = latency.validation_images(dataset, [])
    calls = []
    runtime = synthetic_runtime(lambda: calls.append(1))
    report = latency.measure_images(selected, runtime, lambda: None)
    assert len(calls) == 4  # Three warm-ups, then one measured image.
    assert report["image_count"] == len(report["samples"]) == 1
    assert report["ocr_calls"] == 1
    encoded = json.dumps(report).lower()
    for secret in ("sensitive-reading-fixture", "raw_text", "candidate_text", "bbox", "image_path"):
        assert secret not in encoded
    assert all(report["summary"][stage]["median_ms"] >= 0 for stage in latency.STAGES)


def test_detector_timer_includes_completion_synchronization(dataset):
    selected, _ = latency.validation_images(dataset, [])
    state = {"clock": 0, "pending": False}

    def detect():
        state["pending"] = True

    def synchronize():
        if state["pending"]:
            state["clock"] += 12_000_000
            state["pending"] = False

    report = latency.measure_images(
        selected, synthetic_runtime(detect), synchronize, clock_ns=lambda: state["clock"],
    )
    assert report["summary"]["detect_ms"]["median_ms"] == 12.0
    assert report["summary"]["total_ms"]["median_ms"] == 12.0


def test_unavailable_runtime_fails_without_zero_latency_result(dataset):
    from src.anpr import pipeline
    from src.anpr.runtime_contracts import RuntimeUnavailable

    original_decode = pipeline.decode_image
    selected, _ = latency.validation_images(dataset, [])

    def unavailable():
        raise RuntimeUnavailable("synthetic missing runtime")

    with pytest.raises(latency.BenchmarkError, match="unavailable"):
        latency.measure_images(selected, synthetic_runtime(unavailable), lambda: None)
    assert pipeline.decode_image is original_decode


def test_percentiles_use_recorded_interpolation_and_reject_nonfinite():
    assert latency.summarize([10.0, 20.0, 30.0, 40.0]) == {
        "median_ms": 25.0, "p95_ms": 38.5, "max_ms": 40.0,
    }
    with pytest.raises(latency.BenchmarkError):
        latency.summarize([float("nan")])
