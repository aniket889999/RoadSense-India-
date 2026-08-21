import os
import yaml
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from src.ml.roboflow_yolo_import import (
    parse_roboflow_data_yaml,
    infer_sequence_key,
    build_contiguous_sequence_groups,
    filter_remap_yolo_label,
    validate_safe_output_path,
    run_roboflow_yolo_import,
    get_file_hash
)

def create_synthetic_image(path: str, color=(100, 100, 100), fmt="JPEG"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path, format=fmt)

def setup_mock_roboflow_dataset(root: str, num_images: int = 30, ext="jpg"):
    data_yaml = {
        "nc": 10,
        "names": ["D00", "D01", "D0w0", "D10", "D11", "D20", "D40", "D43", "D44", "D50"],
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "roboflow": {"workspace": "test-ws", "project": "test-proj", "version": 1}
    }
    with open(os.path.join(root, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    splits = ["train", "valid", "test"]
    for i in range(num_images):
        s = splits[i % 3]
        run_id = i // 5
        offset = i % 5
        frame_num = (run_id * 20) + offset
        fname = f"India_{frame_num:06d}_jpg.rf.mockhash{i:04d}.{ext}"
        img_path = os.path.join(root, s, "images", fname)
        lbl_path = os.path.join(root, s, "labels", f"India_{frame_num:06d}_jpg.rf.mockhash{i:04d}.txt")
        fmt = "PNG" if ext == "png" else "JPEG"
        create_synthetic_image(img_path, color=(i % 256, (i * 7) % 256, (i * 13) % 256), fmt=fmt)

        os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
        with open(lbl_path, "w", encoding="utf-8") as lf:
            if i % 3 == 0:
                # Class 6 (D40) and class 0 (D00)
                lf.write("6 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.05 0.05\n")
            elif i % 3 == 1:
                # Only class 5 (D20) -> becomes empty negative
                lf.write("5 0.3 0.3 0.1 0.1\n")
            else:
                # Empty label -> empty negative
                pass

def test_data_yaml_schema_validations(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()

    # nc != 10
    with open(d / "data.yaml", "w") as f:
        yaml.dump({"nc": 999, "names": ["D00"] * 10}, f)
    ok, _, err = parse_roboflow_data_yaml(str(d))
    assert not ok and "must be exactly integer 10" in err

    # dict names with invalid key
    with open(d / "data.yaml", "w") as f:
        yaml.dump({"nc": 10, "names": {"bad_key": "D00", 1: "D01"}}, f)
    ok, _, err = parse_roboflow_data_yaml(str(d))
    assert not ok and "cannot be parsed as integer" in err

    # invalid class sequence
    with open(d / "data.yaml", "w") as f:
        yaml.dump({"nc": 10, "names": ["D00"] * 10}, f)
    ok, _, err = parse_roboflow_data_yaml(str(d))
    assert not ok and "Expected exact 10 source classes" in err

def test_contiguous_sequence_grouping_rule():
    fnames = [
        "India_000008.jpg",
        "India_000009.jpg",
        "India_000010.jpg",
        "India_000020.jpg",
        "India_000021.jpg"
    ]
    mapping, runs, adj_pairs, errs = build_contiguous_sequence_groups(fnames, max_consecutive_gap=1)
    assert not errs
    assert runs == 2
    assert adj_pairs == 3
    assert mapping["India_000008.jpg"] == mapping["India_000009.jpg"] == mapping["India_000010.jpg"]
    assert mapping["India_000020.jpg"] == mapping["India_000021.jpg"]
    assert mapping["India_000009.jpg"] != mapping["India_000020.jpg"]

def test_path_safety_rejections(tmp_path):
    repo_root = tmp_path / "repo"
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True)
    src = tmp_path / "downloads" / "RDD2022-India"
    src.mkdir(parents=True)

    # Regular file target
    reg_file = processed_dir / "regular_file.txt"
    reg_file.touch()
    ok, err = validate_safe_output_path(src, reg_file, repo_root)
    assert not ok and "existing regular file" in err

    # Output == Source
    ok, err = validate_safe_output_path(src, src, repo_root)
    assert not ok and "equal to source root" in err

    # Output inside source
    ok, err = validate_safe_output_path(src, src / "out", repo_root)
    assert not ok and "inside source root" in err

    # Output parent of source
    ok, err = validate_safe_output_path(src / "sub", src, repo_root)
    assert not ok and "parent of source root" in err

    # Output == repo root
    ok, err = validate_safe_output_path(src, repo_root, repo_root)
    assert not ok and "cannot be repository root" in err

    # Output == repo data
    ok, err = validate_safe_output_path(src, repo_root / "data", repo_root)
    assert not ok and "cannot be repo data folder" in err

    # Output == data/processed
    ok, err = validate_safe_output_path(src, processed_dir, repo_root)
    assert not ok and "cannot be data/processed directly" in err

    # Output outside data/processed
    ok, err = validate_safe_output_path(src, tmp_path / "somewhere_else", repo_root)
    assert not ok and "must be a child of" in err

def test_full_import_and_deterministic_fingerprint(tmp_path):
    repo_dir = Path(__file__).resolve().parents[1]
    src = tmp_path / "mock_rf"
    src.mkdir()
    setup_mock_roboflow_dataset(str(src), num_images=30)
    out = repo_dir / "data" / "processed" / f".test_tmp_ds_{os.getpid()}"

    try:
        src_hashes_before = {}
        for root, _, files in os.walk(src):
            for f in files:
                p = os.path.join(root, f)
                src_hashes_before[p] = get_file_hash(p)

        ok1, sum1, err1 = run_roboflow_yolo_import(
            source_root=str(src),
            output_dir=str(out),
            max_consecutive_gap=1,
            seed=42,
            dry_run=True
        )
        assert ok1 is True
        assert not out.exists()

        ok2, sum2, err2 = run_roboflow_yolo_import(
            source_root=str(src),
            output_dir=str(out),
            max_consecutive_gap=1,
            seed=42,
            dry_run=False,
            overwrite=True
        )
        assert ok2 is True
        assert sum1["fingerprint"] == sum2["fingerprint"]
        assert sum2["counts"]["adjacent_numeric_pairs_cross_split"] == 0

        # Source files unmodified
        for p, h in src_hashes_before.items():
            assert get_file_hash(p) == h

        # Check label remapping
        for split in ["train", "val", "test"]:
            lbl_dir = out / "labels" / split
            for f in os.listdir(lbl_dir):
                content = (lbl_dir / f).read_text().strip()
                if content:
                    for line in content.splitlines():
                        assert line.startswith("0 ")

    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)

def test_injected_promotion_failure_rollback(tmp_path):
    repo_dir = Path(__file__).resolve().parents[1]
    src = tmp_path / "mock_rf"
    src.mkdir()
    setup_mock_roboflow_dataset(str(src), num_images=15)
    out = repo_dir / "data" / "processed" / f".test_rollback_prom_{os.getpid()}"

    try:
        out.mkdir(parents=True, exist_ok=True)
        sentinel_file = out / "sentinel.txt"
        sentinel_file.write_text("original_dataset_data")

        ok, summary, err = run_roboflow_yolo_import(
            source_root=str(src),
            output_dir=str(out),
            max_consecutive_gap=1,
            seed=42,
            dry_run=False,
            overwrite=True,
            _inject_promotion_failure=True
        )
        assert ok is False
        assert "Injected promotion failure" in err
        assert out.exists()
        assert (out / "sentinel.txt").read_text() == "original_dataset_data"

    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)

def test_injected_validation_failure_rollback(tmp_path):
    repo_dir = Path(__file__).resolve().parents[1]
    src = tmp_path / "mock_rf"
    src.mkdir()
    setup_mock_roboflow_dataset(str(src), num_images=15)
    out = repo_dir / "data" / "processed" / f".test_rollback_val_{os.getpid()}"

    try:
        out.mkdir(parents=True, exist_ok=True)
        sentinel_file = out / "sentinel.txt"
        sentinel_file.write_text("original_dataset_data")

        ok, summary, err = run_roboflow_yolo_import(
            source_root=str(src),
            output_dir=str(out),
            max_consecutive_gap=1,
            seed=42,
            dry_run=False,
            overwrite=True,
            _inject_validation_failure=True
        )
        assert ok is False
        assert "Injected validation failure" in err
        assert out.exists()
        assert (out / "sentinel.txt").read_text() == "original_dataset_data"

    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)

def test_duplicate_image_same_labels_fails(tmp_path):
    src = tmp_path / "mock_rf"
    src.mkdir()
    setup_mock_roboflow_dataset(str(src), num_images=15)

    # Exact duplicate image content with same labels
    img1 = src / "train" / "images" / "India_000000_jpg.rf.mockhash0000.jpg"
    img2 = src / "valid" / "images" / "India_000099_jpg.rf.mockhash0099.jpg"
    shutil.copy2(img1, img2)
    lbl1 = src / "train" / "labels" / "India_000000_jpg.rf.mockhash0000.txt"
    lbl2 = src / "valid" / "labels" / "India_000099_jpg.rf.mockhash0099.txt"
    shutil.copy2(lbl1, lbl2)

    repo_dir = Path(__file__).resolve().parents[1]
    out = repo_dir / "data" / "processed" / f".test_dup_{os.getpid()}"

    ok, _, err = run_roboflow_yolo_import(str(src), str(out), dry_run=True)
    assert not ok
    assert "Duplicate image content detected" in err
    assert "India_000000_jpg" in err and "India_000099_jpg" in err

def test_nested_source_folders_recursive_discovery(tmp_path):
    src = tmp_path / "mock_rf"
    src.mkdir()
    setup_mock_roboflow_dataset(str(src), num_images=15)

    # Move an image into a subfolder
    sub_img_dir = src / "train" / "images" / "nested_sub"
    sub_lbl_dir = src / "train" / "labels" / "nested_sub"
    sub_img_dir.mkdir(parents=True)
    sub_lbl_dir.mkdir(parents=True)

    img_f = src / "train" / "images" / "India_000000_jpg.rf.mockhash0000.jpg"
    lbl_f = src / "train" / "labels" / "India_000000_jpg.rf.mockhash0000.txt"
    shutil.move(str(img_f), str(sub_img_dir / img_f.name))
    shutil.move(str(lbl_f), str(sub_lbl_dir / lbl_f.name))

    repo_dir = Path(__file__).resolve().parents[1]
    out = repo_dir / "data" / "processed" / f".test_nested_{os.getpid()}"

    try:
        ok, summary, err = run_roboflow_yolo_import(str(src), str(out), max_consecutive_gap=1, dry_run=False, overwrite=True)
        assert ok is True, f"Failed: {err}"
        assert (out / "images" / "train" / "nested_sub" / img_f.name).exists() or (out / "images" / "val" / "nested_sub" / img_f.name).exists() or (out / "images" / "test" / "nested_sub" / img_f.name).exists()
    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
