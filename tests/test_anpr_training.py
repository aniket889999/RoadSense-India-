"""Focused unit tests for dedicated ANPR single-class plate-detector training tools."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import yaml

from scripts.train_anpr_detector import main as train_cli_main
from src.anpr.dataset import DatasetValidationError, prepare_dataset
from src.anpr.training import (
    AnprTrainingError,
    REQUIRED_TRAINING_ARTIFACTS,
    build_training_plan,
    execute_training_plan,
    _assert_trained_model_is_single_plate_class,
    run_anpr_detector_training,
)


def _make_png_bytes(width: int = 120, height: int = 80, color: tuple[int, int, int] = (40, 90, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _create_prepared_anpr_fixture(tmp_path: Path) -> tuple[Path, Path, str, Path]:
    """Create a synthetic repo root with models/, configs/, and a valid prepared ANPR dataset."""
    repo_root = tmp_path / "repo"
    models_dir = repo_root / "models"
    configs_dir = repo_root / "configs" / "training"
    raw_dir = tmp_path / "raw_sources"
    models_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    base_weights_bytes = b"local-yolov8n-base-weights-fixture"
    base_weights_path = models_dir / "yolov8n.pt"
    base_weights_path.write_bytes(base_weights_bytes)
    base_sha256 = hashlib.sha256(base_weights_bytes).hexdigest()

    # Create 3 distinct groups with plates so prepare_dataset assigns train, val, and test
    records = []
    plates = ["MH12AB1234", "DL01CA9999", "KA05MN4321"]
    for idx, plate_text in enumerate(plates, start=1):
        img_name = f"veh_{idx}.png"
        (raw_dir / img_name).write_bytes(_make_png_bytes(120, 80, (idx * 30, 80, 120)))
        records.append(
            {
                "image_id": f"img_{idx}",
                "image_path": img_name,
                "group_id": f"session_{idx}",
                "source_id": f"cam_{idx}",
                "plates": [
                    {
                        "bbox": {"x1": 15.0, "y1": 20.0, "x2": 95.0, "y2": 55.0},
                        "text": plate_text,
                    }
                ],
                "conditions": ["clear"],
            }
        )

    manifest_path = tmp_path / "source_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )

    dataset_dir = repo_root / "data" / "processed" / "anpr_v1"
    # Note: prepare_dataset uses src.anpr.dataset.REPO_ROOT for _safe_output checks,
    # so we monkeypatch REPO_ROOT in tests calling prepare_dataset.
    config_payload = {
        "experiment": {
            "name": "anpr_plate_detector_test_run",
            "seed": 42,
        },
        "model": {
            "base_weights": "models/yolov8n.pt",
            "base_weights_sha256": base_sha256,
            "image_size": 320,
            "class_name": "license_plate",
        },
        "training": {
            "epochs": 5,
            "batch": 2,
            "patience": 3,
            "device": "cpu",
            "workers": 0,
        },
        "augmentation": {
            "fliplr": 0.0,
            "flipud": 0.0,
            "degrees": 3.0,
            "translate": 0.05,
            "scale": 0.15,
            "shear": 0.0,
            "perspective": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "hsv_h": 0.01,
            "hsv_s": 0.25,
            "hsv_v": 0.25,
        },
    }
    config_path = configs_dir / "anpr_plate_detector.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    return repo_root, config_path, base_sha256, dataset_dir


def _prepare_fixture_dataset(monkeypatch, tmp_path: Path) -> tuple[Path, Path, str, Path]:
    import src.anpr.dataset as dataset_mod

    repo_root, config_path, base_sha256, dataset_dir = _create_prepared_anpr_fixture(tmp_path)
    monkeypatch.setattr(dataset_mod, "REPO_ROOT", repo_root)
    monkeypatch.setattr("src.anpr.training._resolve_git_sha", lambda root: "c" * 40)
    prepare_dataset(
        manifest_path=tmp_path / "source_manifest.jsonl",
        source_root=tmp_path / "raw_sources",
        output_dir=dataset_dir,
        seed=42,
        dry_run=False,
    )
    return repo_root, config_path, base_sha256, dataset_dir


def test_importing_training_module_does_not_load_heavy_ml_packages():
    code = (
        "import sys; "
        "import src.anpr.training, scripts.train_anpr_detector; "
        "loaded = [m for m in ('torch', 'ultralytics', 'easyocr') if m in sys.modules]; "
        "assert not loaded, f'Unexpected eager ML imports: {loaded}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_dry_run_validates_inputs_without_loading_yolo_or_creating_outputs(monkeypatch, tmp_path):
    repo_root, config_path, base_sha256, dataset_dir = _prepare_fixture_dataset(monkeypatch, tmp_path)
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)

    summary = run_anpr_detector_training(
        config_path=config_path,
        dataset_dir=dataset_dir,
        execute=False,
        repo_root=repo_root,
    )

    assert "ultralytics" not in sys.modules
    assert summary["mode"] == "dry_run"
    assert summary["run_name"] == "anpr_plate_detector_test_run"
    assert summary["base_weights_sha256"] == base_sha256
    assert summary["class_mapping"] == {"0": "license_plate"}
    assert summary["validation_split"] == "val"
    assert summary["test_evaluation_invoked"] is False
    assert summary["augmentation"]["fliplr"] == 0.0
    assert summary["augmentation"]["flipud"] == 0.0
    assert not (repo_root / "outputs" / "training" / "anpr").exists()


def test_dry_run_rejects_pothole_weights_sha_mismatch_and_existing_run_dir(monkeypatch, tmp_path):
    repo_root, config_path, base_sha256, dataset_dir = _prepare_fixture_dataset(monkeypatch, tmp_path)

    # 1. Pothole weight substitution rejected
    pothole_weights = repo_root / "models" / "pothole_best.pt"
    pothole_weights.write_bytes(b"pothole-checkpoint")
    pothole_sha = hashlib.sha256(b"pothole-checkpoint").hexdigest()
    with pytest.raises(AnprTrainingError, match="pothole"):
        build_training_plan(
            config_path=config_path,
            dataset_dir=dataset_dir,
            base_weights_override="models/pothole_best.pt",
            base_weights_sha256_override=pothole_sha,
            repo_root=repo_root,
        )

    # 2. Wrong SHA-256 rejected
    with pytest.raises(AnprTrainingError, match="SHA-256 mismatch"):
        build_training_plan(
            config_path=config_path,
            dataset_dir=dataset_dir,
            base_weights_sha256_override="0" * 64,
            repo_root=repo_root,
        )

    # 3. Existing run directory rejected to preserve prior outputs
    existing_run = repo_root / "outputs" / "training" / "anpr" / "anpr_plate_detector_test_run"
    existing_run.mkdir(parents=True)
    with pytest.raises(AnprTrainingError, match="already exists"):
        build_training_plan(
            config_path=config_path,
            dataset_dir=dataset_dir,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    "bad_augmentation,expected_match",
    [
        ({"fliplr": 0.5}, "fliplr"),
        ({"flipud": 0.2}, "flipud"),
        ({"perspective": 0.01}, "perspective"),
        ({"copy_paste": 0.3}, "copy_paste"),
        ({"mosaic": 1.0}, "mosaic"),
    ],
)
def test_Rejects_text_mirroring_and_aggressive_augmentations(
    monkeypatch, tmp_path, bad_augmentation, expected_match
):
    repo_root, config_path, _, dataset_dir = _prepare_fixture_dataset(monkeypatch, tmp_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["augmentation"].update(bad_augmentation)
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(AnprTrainingError, match=expected_match):
        build_training_plan(
            config_path=config_path,
            dataset_dir=dataset_dir,
            repo_root=repo_root,
        )


def test_execute_runs_val_only_training_records_provenance_and_never_touches_test_split(
    monkeypatch, tmp_path
):
    repo_root, config_path, base_sha256, dataset_dir = _prepare_fixture_dataset(monkeypatch, tmp_path)
    loaded_weights = []
    train_calls = []

    class FakeYOLO:
        def __init__(self, weights_path: str):
            loaded_weights.append(weights_path)
            self.names = {0: "license_plate"}
            self.trainer = None

        def train(self, **kwargs):
            train_calls.append(kwargs)
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            (weights_dir / "best.pt").write_bytes(b"trained-best-plate-weights")
            (weights_dir / "last.pt").write_bytes(b"trained-last-plate-weights")
            (run_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.85\n", encoding="utf-8")
            (run_dir / "args.yaml").write_text(yaml.safe_dump(kwargs), encoding="utf-8")
            self.trainer = type("Trainer", (), {"save_dir": run_dir})()
            return self.trainer

        def val(self, *args, **kwargs):
            raise AssertionError("Training wrapper must never invoke separate val()/test evaluation.")

    metadata = run_anpr_detector_training(
        config_path=config_path,
        dataset_dir=dataset_dir,
        execute=True,
        repo_root=repo_root,
        yolo_loader=FakeYOLO,
    )

    assert loaded_weights == [str(repo_root / "models" / "yolov8n.pt")]
    assert len(train_calls) == 1
    call = train_calls[0]
    assert call["data"] == str(dataset_dir / "anpr.yaml")
    assert call["project"] == str(repo_root / "outputs" / "training" / "anpr")
    assert call["name"] == "anpr_plate_detector_test_run"
    assert call["exist_ok"] is False
    assert call["val"] is True
    assert call["split"] == "val"
    assert call["amp"] is False
    assert call["fliplr"] == 0.0
    assert call["flipud"] == 0.0
    assert call["mosaic"] == 0.0
    assert call["copy_paste"] == 0.0

    run_dir = repo_root / "outputs" / "training" / "anpr" / "anpr_plate_detector_test_run"
    metadata_file = run_dir / "model_metadata.json"
    assert metadata_file.is_file()
    saved_meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert saved_meta == metadata
    assert saved_meta["class_mapping"] == {"0": "license_plate"}
    assert saved_meta["base_weights_sha256"] == base_sha256
    assert saved_meta["validation_split"] == "val"
    assert saved_meta["test_evaluation_invoked"] is False
    assert len(saved_meta["dataset_fingerprint"]) == 64
    assert set(saved_meta["artifacts"].keys()) == set(REQUIRED_TRAINING_ARTIFACTS)
    for digest in saved_meta["artifacts"].values():
        assert len(digest) == 64


def test_execute_fails_explicitly_when_required_training_artifact_is_missing(
    monkeypatch, tmp_path
):
    repo_root, config_path, _, dataset_dir = _prepare_fixture_dataset(monkeypatch, tmp_path)

    class IncompleteFakeYOLO:
        def __init__(self, weights_path: str):
            self.names = {0: "license_plate"}
            self.trainer = None

        def train(self, **kwargs):
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            # Intentionally omit weights/best.pt
            (weights_dir / "last.pt").write_bytes(b"only-last")
            (run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
            (run_dir / "args.yaml").write_text("{}", encoding="utf-8")
            self.trainer = type("Trainer", (), {"save_dir": run_dir})()
            return self.trainer

    with pytest.raises(AnprTrainingError, match="weights/best.pt"):
        run_anpr_detector_training(
            config_path=config_path,
            dataset_dir=dataset_dir,
            execute=True,
            repo_root=repo_root,
            yolo_loader=IncompleteFakeYOLO,
        )


def test_cli_dry_run_and_invalid_dataset_error_handling(monkeypatch, tmp_path, capsys):
    repo_root, config_path, _, dataset_dir = _prepare_fixture_dataset(monkeypatch, tmp_path)
    import src.anpr.training as training_mod

    monkeypatch.setattr(training_mod, "REPO_ROOT", repo_root)

    exit_code = train_cli_main(
        [
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_dir),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["mode"] == "dry_run"
    assert payload["class_mapping"] == {"0": "license_plate"}

    # Missing/invalid dataset returns exit code 1
    bad_exit = train_cli_main(
        [
            "--config",
            str(config_path),
            "--dataset",
            str(repo_root / "data" / "processed" / "missing_dataset"),
            "--dry-run",
        ]
    )
    assert bad_exit == 1


@pytest.mark.parametrize("names", [None, {0.5: "license_plate"}, {0: "pothole"}])
def test_training_requires_the_same_class_schema_as_the_runtime(names):
    model = type("Model", (), {"names": names})()
    with pytest.raises(AnprTrainingError, match="class verification"):
        _assert_trained_model_is_single_plate_class(model)


def test_config_change_after_preflight_is_rejected_before_loading_yolo(monkeypatch, tmp_path):
    repo_root, config, _, dataset = _prepare_fixture_dataset(monkeypatch, tmp_path)
    plan = build_training_plan(config_path=config, dataset_dir=dataset, repo_root=repo_root)
    config.write_text(config.read_text() + "\n# changed after audit\n")
    loads = []
    with pytest.raises(AnprTrainingError, match="configuration changed"):
        execute_training_plan(plan, yolo_loader=lambda path: loads.append(path))
    assert loads == []
