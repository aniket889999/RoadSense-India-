import os
import sys
import yaml
import json
import math
import shutil
import hashlib
import tempfile
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.ml.metrics import extract_detection_metrics
from src.ml.metadata import build_training_metadata

def create_sha256(content: bytes = b"test") -> str:
    return hashlib.sha256(content).hexdigest()

def test_full_training_config_uses_mps_not_auto():
    cfg_path = REPO_ROOT / "configs" / "training" / "pothole_yolov8n_rdd2022_india.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["training"]["device"] == "mps", "Full training config must use 'device: mps'"
    assert "auto" not in cfg["training"]["device"]
    assert cfg["experiment"]["name"] == "pothole_yolov8n_rdd2022_india_mps_baseline_v1"
    assert cfg["model"]["base_weights"] == "models/yolov8n.pt"

def test_evaluate_cli_requires_device_batch_imgsz():
    eval_script = REPO_ROOT / "scripts" / "evaluate_pothole.py"

    # Missing --device, --batch, --imgsz
    res = subprocess.run([
        sys.executable, str(eval_script),
        "--weights", "outputs/training/pothole_yolov8n_smoke/weights/best.pt",
        "--dataset", "data/processed/rdd2022_india_roboflow_d40_v1"
    ], capture_output=True, text=True)
    assert res.returncode != 0
    assert "--device" in res.stderr or "required" in res.stderr

def setup_mock_training_and_dataset(tmp_path):
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifests").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img = Image.new("RGB", (32, 32), color=(120, 120, 120))
        img_path = img_dir / f"img_001_{split}.jpg"
        img.save(img_path)
        lbl_path = lbl_dir / f"img_001_{split}.txt"
        lbl_path.write_text("0 0.5 0.5 0.2 0.2\n")

    pothole_yaml = dataset_dir / "pothole.yaml"
    yaml_content = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "pothole"}
    }
    with open(pothole_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)

    fingerprint = create_sha256(b"fingerprint_dataset")
    prep_meta = {
        "dataset_fingerprint": fingerprint,
        "counts": {
            "split_distribution": {
                "train": {"images": 1, "positive_images": 1, "d40_instances": 1},
                "val": {"images": 1, "positive_images": 1, "d40_instances": 1},
                "test": {"images": 1, "positive_images": 1, "d40_instances": 1}
            }
        }
    }
    with open(dataset_dir / "manifests" / "preparation_metadata.json", "w") as f:
        json.dump(prep_meta, f)

    # Base weights
    base_weights_path = tmp_path / "models" / "yolov8n.pt"
    base_weights_path.parent.mkdir(parents=True, exist_ok=True)
    base_weights_path.write_bytes(b"dummy_base_weights")
    base_weights_hash = hashlib.sha256(b"dummy_base_weights").hexdigest()

    # Mock Training Run strictly inside outputs/training
    run_dir = tmp_path / "outputs" / "training" / "mock_exp_run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_pt = weights_dir / "best.pt"
    best_pt.write_bytes(b"dummy_best_weights_123")
    best_pt_hash = hashlib.sha256(b"dummy_best_weights_123").hexdigest()

    last_pt = weights_dir / "last.pt"
    last_pt.write_bytes(b"dummy_last_weights")
    last_pt_hash = hashlib.sha256(b"dummy_last_weights").hexdigest()

    results_csv = run_dir / "results.csv"
    results_csv.write_text("epoch,loss\n1,0.5\n")
    results_csv_hash = hashlib.sha256(results_csv.read_bytes()).hexdigest()

    args_yaml = run_dir / "args.yaml"
    args_yaml.write_text("epochs: 50\n")
    args_yaml_hash = hashlib.sha256(args_yaml.read_bytes()).hexdigest()

    train_config = run_dir / "train_config.yaml"
    train_config.write_text("experiment:\n  name: mock_exp_run\n")
    train_config_hash = hashlib.sha256(train_config.read_bytes()).hexdigest()

    dataset_prep_meta = run_dir / "dataset_preparation_metadata.json"
    dataset_prep_meta.write_text(json.dumps(prep_meta))
    dataset_prep_meta_hash = hashlib.sha256(dataset_prep_meta.read_bytes()).hexdigest()

    model_meta = {
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        "dataset_fingerprint": fingerprint,
        "run_directory": str(run_dir.resolve()),
        "resolved_base_weights": str(base_weights_path.resolve()),
        "base_weights": str(base_weights_path.resolve()),
        "artifacts": {
            "base_weights": base_weights_hash,
            "weights/best.pt": best_pt_hash,
            "weights/last.pt": last_pt_hash,
            "results.csv": results_csv_hash,
            "args.yaml": args_yaml_hash,
            "train_config.yaml": train_config_hash,
            "dataset_preparation_metadata.json": dataset_prep_meta_hash
        }
    }
    with open(run_dir / "model_metadata.json", "w") as f:
        json.dump(model_meta, f)

    return dataset_dir, best_pt, fingerprint

def test_mocked_evaluation_one_time_and_explicit_arguments(tmp_path, monkeypatch):
    dataset_dir, best_pt, fingerprint = setup_mock_training_and_dataset(tmp_path)

    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    mock_val_results = MagicMock()
    mock_val_results.results_dict = {
        "metrics/precision(B)": 0.82,
        "metrics/recall(B)": 0.75,
        "metrics/mAP50(B)": 0.80,
        "metrics/mAP50-95(B)": 0.52
    }

    mock_model_instance = MagicMock()
    mock_model_instance.val.return_value = mock_val_results
    mock_yolo_cls = MagicMock(return_value=mock_model_instance)

    with patch("ultralytics.YOLO", mock_yolo_cls):
        test_args = [
            "evaluate_pothole.py",
            "--weights", str(best_pt),
            "--dataset", str(dataset_dir),
            "--device", "mps",
            "--batch", "4",
            "--imgsz", "640"
        ]
        monkeypatch.setattr("sys.argv", test_args)

        # 1. First execution succeeds and reserves folder
        ep.main()

        mock_model_instance.val.assert_called_once_with(
            data=str(dataset_dir / "pothole.yaml"),
            split="test",
            device="mps",
            batch=4,
            imgsz=640,
            project=str(tmp_path / "outputs" / "evaluation"),
            name="mock_exp_run",
            exist_ok=True
        )

        eval_dir = tmp_path / "outputs" / "evaluation" / "mock_exp_run"
        assert (eval_dir / "evaluation_attempt.json").exists()
        assert (eval_dir / "test_metrics.json").exists()

        with open(eval_dir / "test_metrics.json") as f:
            report = json.load(f)
        assert report["metrics"]["precision"] == 0.82
        assert report["dataset_fingerprint"] == fingerprint
        assert report["evaluation_params"]["device"] == "mps"

        # 2. Second execution is blocked by reserved directory without calling YOLO or model.val()
        mock_model_instance.val.reset_mock()
        mock_yolo_cls.reset_mock()
        with pytest.raises(SystemExit) as exc:
            ep.main()
        assert exc.value.code == 1
        mock_yolo_cls.assert_not_called()
        mock_model_instance.val.assert_not_called()

def test_external_training_run_rejected_before_yolo(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    # Move weights outside outputs/training
    outside_run = tmp_path / "external_run" / "weights"
    outside_run.mkdir(parents=True)
    outside_best = outside_run / "best.pt"
    outside_best.write_bytes(b"dummy_weights")

    mock_yolo_cls = MagicMock()
    with patch("ultralytics.YOLO", mock_yolo_cls):
        test_args = [
            "evaluate_pothole.py",
            "--weights", str(outside_best),
            "--dataset", str(dataset_dir),
            "--device", "mps",
            "--batch", "4",
            "--imgsz", "640"
        ]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit) as exc:
            ep.main()
        assert exc.value.code == 1
        mock_yolo_cls.assert_not_called()

def test_metadata_run_directory_mismatch_rejected(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    run_dir = best_pt.parent.parent
    with open(run_dir / "model_metadata.json", "r") as f:
        meta = json.load(f)
    meta["run_directory"] = str(tmp_path / "outputs" / "training" / "other_run")
    with open(run_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f)

    test_args = [
        "evaluate_pothole.py",
        "--weights", str(best_pt),
        "--dataset", str(dataset_dir),
        "--device", "mps",
        "--batch", "4",
        "--imgsz", "640"
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit) as exc:
        ep.main()
    assert exc.value.code == 1

def test_tampered_best_pt_rejected(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    # Tamper with best.pt on disk
    best_pt.write_bytes(b"tampered_best_pt_bytes")

    test_args = [
        "evaluate_pothole.py",
        "--weights", str(best_pt),
        "--dataset", str(dataset_dir),
        "--device", "mps",
        "--batch", "4",
        "--imgsz", "640"
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit) as exc:
        ep.main()
    assert exc.value.code == 1

def test_symlinked_outputs_training_pointing_outside_rejected(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    outside_dir = tmp_path.parent / f"outside_training_{os.getpid()}"
    outside_dir.mkdir(parents=True, exist_ok=True)

    outputs_training = tmp_path / "outputs" / "training"
    shutil.rmtree(outputs_training)
    outputs_training.symlink_to(outside_dir, target_is_directory=True)

    try:
        test_args = [
            "evaluate_pothole.py",
            "--weights", str(best_pt),
            "--dataset", str(dataset_dir),
            "--device", "mps",
            "--batch", "4",
            "--imgsz", "640"
        ]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit) as exc:
            ep.main()
        assert exc.value.code == 1
    finally:
        if outside_dir.exists():
            shutil.rmtree(outside_dir, ignore_errors=True)

def test_symlinked_outputs_evaluation_pointing_outside_rejected(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    outside_dir = tmp_path.parent / f"outside_eval_{os.getpid()}"
    outside_dir.mkdir(parents=True, exist_ok=True)

    outputs_eval = tmp_path / "outputs" / "evaluation"
    outputs_eval.symlink_to(outside_dir, target_is_directory=True)

    try:
        test_args = [
            "evaluate_pothole.py",
            "--weights", str(best_pt),
            "--dataset", str(dataset_dir),
            "--device", "mps",
            "--batch", "4",
            "--imgsz", "640"
        ]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit) as exc:
            ep.main()
        assert exc.value.code == 1
    finally:
        if outside_dir.exists():
            shutil.rmtree(outside_dir, ignore_errors=True)

def test_symlinked_model_metadata_rejected(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    run_dir = best_pt.parent.parent
    real_meta = run_dir / "real_meta.json"
    (run_dir / "model_metadata.json").rename(real_meta)
    (run_dir / "model_metadata.json").symlink_to(real_meta)

    test_args = [
        "evaluate_pothole.py",
        "--weights", str(best_pt),
        "--dataset", str(dataset_dir),
        "--device", "mps",
        "--batch", "4",
        "--imgsz", "640"
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit) as exc:
        ep.main()
    assert exc.value.code == 1

def test_symlinked_best_pt_file_rejected(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    # Symlink best.pt directly
    real_pt = best_pt.parent / "real_best.pt"
    best_pt.rename(real_pt)
    best_pt.symlink_to(real_pt)

    mock_yolo_cls = MagicMock()
    with patch("ultralytics.YOLO", mock_yolo_cls):
        test_args = [
            "evaluate_pothole.py",
            "--weights", str(best_pt),
            "--dataset", str(dataset_dir),
            "--device", "mps",
            "--batch", "4",
            "--imgsz", "640"
        ]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit) as exc:
            ep.main()
        assert exc.value.code == 1
        mock_yolo_cls.assert_not_called()

def test_symlinked_weights_directory_rejected(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    run_dir = best_pt.parent.parent
    visible_weights_dir = run_dir / "weights"
    real_weights_dir = run_dir / "real_weights_target"
    visible_weights_dir.rename(real_weights_dir)
    visible_weights_dir.symlink_to(real_weights_dir, target_is_directory=True)

    weights_param = visible_weights_dir / "best.pt"

    mock_yolo_cls = MagicMock()
    with patch("ultralytics.YOLO", mock_yolo_cls):
        test_args = [
            "evaluate_pothole.py",
            "--weights", str(weights_param),
            "--dataset", str(dataset_dir),
            "--device", "mps",
            "--batch", "4",
            "--imgsz", "640"
        ]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit) as exc:
            ep.main()
        assert exc.value.code == 1
        mock_yolo_cls.assert_not_called()

        eval_dir = tmp_path / "outputs" / "evaluation" / "mock_exp_run"
        assert not eval_dir.exists()

def test_invalid_cli_arguments_fail_without_reserving_dir(tmp_path, monkeypatch):
    dataset_dir, best_pt, _ = setup_mock_training_and_dataset(tmp_path)
    import scripts.evaluate_pothole as ep
    monkeypatch.setattr(ep, "REPO_ROOT", tmp_path)

    invalid_param_sets = [
        {"device": "  ", "batch": 4, "imgsz": 640},
        {"device": "mps", "batch": 0, "imgsz": 640},
        {"device": "mps", "batch": -1, "imgsz": 640},
        {"device": "mps", "batch": 4, "imgsz": 0},
        {"device": "mps", "batch": 4, "imgsz": -320}
    ]

    for params in invalid_param_sets:
        test_args = [
            "evaluate_pothole.py",
            "--weights", str(best_pt),
            "--dataset", str(dataset_dir),
            "--device", params["device"],
            "--batch", str(params["batch"]),
            "--imgsz", str(params["imgsz"])
        ]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit) as exc:
            ep.main()
        assert exc.value.code == 1

        eval_dir = tmp_path / "outputs" / "evaluation" / "mock_exp_run"
        assert not eval_dir.exists(), f"Evaluation directory was created unexpectedly for invalid params: {params}"

def test_mocked_train_pothole_success_and_missing_artifact(tmp_path, monkeypatch):
    import scripts.train_pothole as tp
    monkeypatch.setattr(tp, "REPO_ROOT", tmp_path)

    dataset_dir, _, _ = setup_mock_training_and_dataset(tmp_path)
    cfg_file = tmp_path / "train_cfg.yaml"
    cfg_data = {
        "experiment": {"name": "test_train_run", "seed": 42},
        "model": {"base_weights": "models/yolov8n.pt", "image_size": 320},
        "training": {"epochs": 1, "batch": 2, "patience": 0, "device": "mps"}
    }
    with open(cfg_file, "w") as f:
        yaml.dump(cfg_data, f)

    run_dir = tmp_path / "outputs" / "training" / "test_train_run"
    weights_dir = run_dir / "weights"

    def populate_artifacts(skip_key=None):
        weights_dir.mkdir(parents=True, exist_ok=True)
        if skip_key != "best.pt":
            (weights_dir / "best.pt").write_bytes(b"best_w")
        if skip_key != "last.pt":
            (weights_dir / "last.pt").write_bytes(b"last_w")
        if skip_key != "results.csv":
            (run_dir / "results.csv").write_text("epoch,loss\n1,0.1\n")
        if skip_key != "args.yaml":
            (run_dir / "args.yaml").write_text("device: mps\n")

    # 1. Missing required artifact in training run fails
    populate_artifacts(skip_key="last.pt")

    mock_trainer = MagicMock(save_dir=str(run_dir))
    mock_model_instance = MagicMock(trainer=mock_trainer)
    mock_yolo_cls = MagicMock(return_value=mock_model_instance)

    with patch("ultralytics.YOLO", mock_yolo_cls):
        test_args = [
            "train_pothole.py",
            "--config", str(cfg_file),
            "--dataset", str(dataset_dir)
        ]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit) as exc:
            tp.main()
        assert exc.value.code == 1
        assert not (run_dir / "model_metadata.json").exists()

    # 2. Complete artifacts succeed and write metadata with all 7 hashes
    populate_artifacts()
    with patch("ultralytics.YOLO", mock_yolo_cls):
        test_args = [
            "train_pothole.py",
            "--config", str(cfg_file),
            "--dataset", str(dataset_dir)
        ]
        monkeypatch.setattr("sys.argv", test_args)

        tp.main()

        meta_path = run_dir / "model_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["resolved_base_weights"] == str((tmp_path / "models" / "yolov8n.pt").resolve())
        assert len(meta["artifacts"]) == 7
        for k in [
            "base_weights", "weights/best.pt", "weights/last.pt", "results.csv",
            "args.yaml", "train_config.yaml", "dataset_preparation_metadata.json"
        ]:
            assert k in meta["artifacts"]
            assert len(meta["artifacts"][k]) == 64

def test_mocked_train_allow_weight_download(tmp_path, monkeypatch):
    import scripts.train_pothole as tp
    monkeypatch.setattr(tp, "REPO_ROOT", tmp_path)

    dataset_dir, _, _ = setup_mock_training_and_dataset(tmp_path)
    # Remove local models/yolov8n.pt
    (tmp_path / "models" / "yolov8n.pt").unlink()

    cfg_file = tmp_path / "train_cfg.yaml"
    cfg_data = {
        "experiment": {"name": "test_dl_run", "seed": 42},
        "model": {"base_weights": "models/yolov8n.pt", "image_size": 320},
        "training": {"epochs": 1, "batch": 2, "patience": 0, "device": "mps"}
    }
    with open(cfg_file, "w") as f:
        yaml.dump(cfg_data, f)

    run_dir = tmp_path / "outputs" / "training" / "test_dl_run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_bytes(b"best_w")
    (weights_dir / "last.pt").write_bytes(b"last_w")
    (run_dir / "results.csv").write_text("epoch,loss\n1,0.1\n")
    (run_dir / "args.yaml").write_text("device: mps\n")

    # Mock downloaded file created by Ultralytics in working directory
    dl_file = tmp_path / "yolov8n.pt"
    dl_file.write_bytes(b"downloaded_yolov8n_bytes")

    mock_trainer = MagicMock(save_dir=str(run_dir))
    mock_model_instance = MagicMock(trainer=mock_trainer, ckpt_path=str(dl_file))
    mock_yolo_cls = MagicMock(return_value=mock_model_instance)

    with patch("ultralytics.YOLO", mock_yolo_cls):
        test_args = [
            "train_pothole.py",
            "--config", str(cfg_file),
            "--dataset", str(dataset_dir),
            "--allow-weight-download"
        ]
        monkeypatch.setattr("sys.argv", test_args)

        tp.main()

        meta_path = run_dir / "model_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["resolved_base_weights"] == str(dl_file.resolve())
        assert meta["artifacts"]["base_weights"] == hashlib.sha256(b"downloaded_yolov8n_bytes").hexdigest()
