"""Focused tests for batch ANPR prediction generation and provenance recording."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.generate_anpr_predictions import (
    BatchPredictionError,
    generate_split_predictions,
    main as generate_cli_main,
)
from src.anpr.contracts import PlateBox
from src.anpr.dataset import prepare_dataset
from src.anpr.evaluation import load_predictions
from src.anpr.runtime_contracts import AnprRuntime, OcrRead, PlateDetection


def _make_png(width: int = 120, height: int = 80, seed_idx: int = 1) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(seed_idx * 35, 100, 150)).save(buf, format="PNG")
    return buf.getvalue()


def _create_prepared_dataset_fixture(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    import src.anpr.dataset as dataset_mod

    repo_root = tmp_path / "repo"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (repo_root / "outputs").mkdir(parents=True)

    records = []
    for idx, plate in enumerate(["MH12AB1234", "DL01CA9999", "KA05MN4321"], start=1):
        fname = f"img_{idx}.png"
        (raw_dir / fname).write_bytes(_make_png(seed_idx=idx))
        records.append(
            {
                "image_id": f"veh_{idx}",
                "image_path": fname,
                "group_id": f"grp_{idx}",
                "source_id": f"src_{idx}",
                "plates": [
                    {
                        "bbox": {"x1": 10.0, "y1": 15.0, "x2": 90.0, "y2": 50.0},
                        "text": plate,
                        "label_provenance": "publisher_mixed",
                    }
                ],
                "conditions": ["other"],
            }
        )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    dataset_dir = repo_root / "data" / "processed" / "anpr_fixture"
    monkeypatch.setattr(dataset_mod, "REPO_ROOT", repo_root)
    prepare_dataset(
        manifest_path=manifest_path,
        source_root=raw_dir,
        output_dir=dataset_dir,
        seed=42,
        dry_run=False,
    )
    return repo_root, dataset_dir


def test_generate_split_predictions_writes_val_predictions_and_provenance_and_blocks_test_split(
    monkeypatch, tmp_path
):
    repo_root, dataset_dir = _create_prepared_dataset_fixture(monkeypatch, tmp_path)

    class FakeDetector:
        checkpoint_path = repo_root / "models" / "anpr" / "fake.pt"
        checkpoint_sha256 = "a" * 64
        device = "cpu"
        confidence_threshold = 0.25
        iou_threshold = 0.45
        image_size = 640

        def detect(self, image_bgr):
            return [
                PlateDetection(
                    bbox=PlateBox(x1=10.0, y1=15.0, x2=90.0, y2=50.0),
                    confidence=0.95,
                )
            ]

    class FakeRecognizer:
        model_dir = repo_root / "models" / "easyocr"
        device = "cpu"
        languages = ("en",)
        weight_files = ()

        def recognize(self, plate_bgr):
            return OcrRead(raw_text="DL 01-CA 9999", confidence=0.88)

    fake_runtime = AnprRuntime(detector=FakeDetector(), recognizer=FakeRecognizer())

    # Held-out test split is refused by default
    with pytest.raises(BatchPredictionError, match="held-out 'test' split"):
        generate_split_predictions(
            dataset_dir=dataset_dir,
            output_jsonl=repo_root / "outputs" / "anpr" / "test_preds.jsonl",
            split="test",
            allow_test_split=False,
            runtime_factory=lambda: fake_runtime,
            repo_root=repo_root,
        )

    out_jsonl = repo_root / "outputs" / "anpr" / "val_preds.jsonl"
    provenance = generate_split_predictions(
        dataset_dir=dataset_dir,
        output_jsonl=out_jsonl,
        split="val",
        runtime_factory=lambda: fake_runtime,
        repo_root=repo_root,
    )

    assert out_jsonl.is_file()
    prov_file = out_jsonl.with_suffix(".provenance.json")
    assert prov_file.is_file()
    assert provenance["split"] == "val"
    assert provenance["images_processed"] == 1
    assert provenance["result_status_counts"]["COMPLETE"] == 1

    loaded = load_predictions(out_jsonl)
    assert len(loaded) == 1
    only_res = next(iter(loaded.values()))
    assert only_res.status == "COMPLETE"
    assert only_res.plates[0].normalized_text == "DL01CA9999"

    # Existing output file is refused
    with pytest.raises(BatchPredictionError, match="already exists"):
        generate_split_predictions(
            dataset_dir=dataset_dir,
            output_jsonl=out_jsonl,
            split="val",
            runtime_factory=lambda: fake_runtime,
            repo_root=repo_root,
        )


def test_generate_cli_rejects_escape_outside_outputs(monkeypatch, tmp_path):
    repo_root, dataset_dir = _create_prepared_dataset_fixture(monkeypatch, tmp_path)
    import scripts.generate_anpr_predictions as gen_mod

    monkeypatch.setattr(gen_mod, "REPO_ROOT", repo_root)
    rc = generate_cli_main(
        [
            "--dataset",
            str(dataset_dir),
            "--split",
            "val",
            "--output",
            str(tmp_path / "outside.jsonl"),
        ]
    )
    assert rc == 1
