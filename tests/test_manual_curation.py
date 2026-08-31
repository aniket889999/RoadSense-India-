import csv
import hashlib
import importlib.util
import json
import os
import stat
import sys
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import pytest
from PIL import Image

import src.ml.manual_curation as curation
from src.ml.manual_curation import ManualCurationError, prepare_manual_curation_batch


MANUAL_HEADER = "incident_id,frame_index,x_min,y_min,x_max,y_max,label,note"
REVIEW_HEADER = "frame_index,review_status,note"


def _make_jpeg(width=100, height=80, *, include_exif=False):
    image = Image.new("RGB", (width, height), color=(31, 95, 151))
    buffer = BytesIO()
    if include_exif:
        exif = Image.Exif()
        exif[270] = "/Users/private-user/very-sensitive-road-note"
        image.save(buffer, format="JPEG", quality=92, exif=exif)
    else:
        image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _make_annotation_kit(frames=None, *, manifest_dimensions=None, extra_members=None):
    """Build a small ZIP compatible with ``src.video_io.process_and_create_kit``."""

    frames = frames or {
        0: _make_jpeg(),
        5: _make_jpeg(),
        10: _make_jpeg(),
    }
    dimensions = manifest_dimensions or (100, 80)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", "synthetic RoadSense annotation kit")
        archive.writestr("manual_potholes_template.csv", MANUAL_HEADER + "\n")
        rows = ["frame_index,timestamp_seconds,frame_file,width,height"]
        for frame_index, jpeg in sorted(frames.items()):
            frame_name = f"frame_{frame_index:05d}.jpg"
            archive.writestr(f"frames/{frame_name}", jpeg)
            rows.append(
                f"{frame_index},{frame_index / 10:.3f},{frame_name},{dimensions[0]},{dimensions[1]}"
            )
        archive.writestr("frame_manifest.csv", "\n".join(rows) + "\n")
        for name, value in (extra_members or {}).items():
            archive.writestr(name, value)
    return buffer.getvalue()


def _manual_csv(rows=()):
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(MANUAL_HEADER.split(","))
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _review_csv(rows):
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(REVIEW_HEADER.split(","))
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _repo(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    return repo_root


def _valid_inputs(*, include_exif=False):
    frames = {
        0: _make_jpeg(include_exif=include_exif),
        5: _make_jpeg(),
        10: _make_jpeg(),
    }
    return (
        _make_annotation_kit(frames),
        _manual_csv([["POT-001", "0", "10", "20", "50", "60", "pothole", "manual note"]]),
        _review_csv(
            [
                ["0", "pothole_confirmed", "/Users/private-user/review note must not export"],
                ["5", "no_pothole_confirmed", "no pothole found"],
            ]
        ),
    )


def _run_valid_batch(repo_root, *, write, overwrite=False, include_exif=False):
    kit, annotations, reviews = _valid_inputs(include_exif=include_exif)
    return prepare_manual_curation_batch(
        kit,
        annotations,
        reviews,
        "rural-road-session-001",
        output_dir="data/interim/manual_curation/rural-road-session-001",
        write=write,
        overwrite=overwrite,
        repo_root=repo_root,
    )


def test_human_reviewed_batch_exports_only_selected_frames_and_yolo_labels(tmp_path):
    repo_root = _repo(tmp_path)

    summary, errors = _run_valid_batch(repo_root, write=True, include_exif=True)

    assert errors == []
    assert summary["dry_run"] is False
    assert summary["reviewed_frame_count"] == 2
    assert summary["pothole_confirmed_frame_count"] == 1
    assert summary["no_pothole_confirmed_frame_count"] == 1
    assert summary["manual_pothole_box_count"] == 1
    assert summary["artifacts"] == [
        "images/",
        "labels/",
        "manifests/curation_manifest.csv",
        "manifests/curation_metadata.json",
    ]

    target = repo_root / "data/interim/manual_curation/rural-road-session-001"
    assert (target / "images/frame_00000.jpg").is_file()
    assert (target / "images/frame_00005.jpg").is_file()
    assert not (target / "images/frame_00010.jpg").exists()
    assert (target / "labels/frame_00000.txt").read_text() == (
        "0 0.3000000000 0.5000000000 0.4000000000 0.5000000000\n"
    )
    assert (target / "labels/frame_00005.txt").read_bytes() == b""
    assert not (target / "labels/frame_00010.txt").exists()

    metadata = json.loads((target / "manifests/curation_metadata.json").read_text())
    assert metadata["human_reviewed"] is True
    assert metadata["training_or_evaluation_performed"] is False
    assert metadata["class_mapping"] == {"0": "pothole"}
    assert metadata["counts"] == {
        "reviewed_frames": 2,
        "pothole_confirmed_frames": 1,
        "no_pothole_confirmed_frames": 1,
        "manual_pothole_boxes": 1,
    }

    exported = target / "images/frame_00000.jpg"
    with Image.open(exported) as output_image:
        assert output_image.size == (100, 80)
        assert dict(output_image.getexif()) == {}

    manifest = (target / "manifests/curation_manifest.csv").read_text()
    serialized_metadata = json.dumps(metadata, sort_keys=True)
    assert "/Users/private-user" not in manifest
    assert "/Users/private-user" not in serialized_metadata
    assert "manual note" not in manifest
    assert "review note" not in manifest
    assert "source_image_sha256" in manifest
    assert "exported_image_sha256" in manifest

    for directory in (target, target / "images", target / "labels", target / "manifests"):
        assert stat.S_IMODE(os.lstat(directory).st_mode) & 0o077 == 0
    for file_path in target.rglob("*"):
        if file_path.is_file():
            assert stat.S_IMODE(os.lstat(file_path).st_mode) & 0o077 == 0
    assert not any(path.suffix.lower() == ".mp4" for path in target.rglob("*"))


def test_dry_run_is_write_free_even_when_curation_parent_does_not_exist(tmp_path):
    repo_root = _repo(tmp_path)
    kit, annotations, reviews = _valid_inputs()
    hashes_before = [hashlib.sha256(item).hexdigest() for item in (kit, annotations, reviews)]

    summary, errors = prepare_manual_curation_batch(
        kit,
        annotations,
        reviews,
        "dry-run-recording",
        output_dir="data/interim/manual_curation/dry-run-recording",
        write=False,
        repo_root=repo_root,
    )

    assert errors == []
    assert summary["dry_run"] is True
    assert not (repo_root / "data").exists()
    assert hashes_before == [hashlib.sha256(item).hexdigest() for item in (kit, annotations, reviews)]


def test_canonical_empty_manual_csv_is_allowed_only_for_explicit_negatives(tmp_path):
    repo_root = _repo(tmp_path)
    kit = _make_annotation_kit()
    empty_annotations = _manual_csv()

    summary, errors = prepare_manual_curation_batch(
        kit,
        empty_annotations,
        _review_csv([["5", "no_pothole_confirmed", "reviewed"]]),
        "negative-only",
        output_dir="data/interim/manual_curation/negative-only",
        write=True,
        repo_root=repo_root,
    )
    assert errors == []
    assert summary["manual_pothole_box_count"] == 0
    assert (repo_root / "data/interim/manual_curation/negative-only/labels/frame_00005.txt").read_bytes() == b""

    _, errors = prepare_manual_curation_batch(
        kit,
        empty_annotations,
        _review_csv([["0", "pothole_confirmed", "reviewed"]]),
        "positive-needs-box",
        output_dir="data/interim/manual_curation/positive-needs-box",
        write=False,
        repo_root=repo_root,
    )
    assert errors
    assert "has no manual pothole box" in errors[0]


@pytest.mark.parametrize(
    ("manual_rows", "review_rows", "message"),
    [
        ([], [["0", "pothole_confirmed", "reviewed"]], "has no manual pothole box"),
        (
            [["POT-001", "0", "10", "20", "50", "60", "pothole", ""]],
            [["0", "no_pothole_confirmed", "reviewed"]],
            "but has manual pothole box",
        ),
        (
            [["POT-001", "0", "10", "20", "50", "60", "pothole", ""]],
            [["5", "no_pothole_confirmed", "reviewed"]],
            "without a frame review",
        ),
    ],
)
def test_human_review_and_manual_boxes_must_agree(tmp_path, manual_rows, review_rows, message):
    repo_root = _repo(tmp_path)
    _, errors = prepare_manual_curation_batch(
        _make_annotation_kit(),
        _manual_csv(manual_rows),
        _review_csv(review_rows),
        "agreement-check",
        output_dir="data/interim/manual_curation/agreement-check",
        write=False,
        repo_root=repo_root,
    )
    assert errors
    assert message in errors[0]
    assert not (repo_root / "data").exists()


@pytest.mark.parametrize(
    ("review_bytes", "message"),
    [
        (
            _review_csv([["999", "no_pothole_confirmed", "unknown"]]),
            "is not in the annotation-kit manifest",
        ),
        (
            _review_csv(
                [
                    ["0", "pothole_confirmed", "one"],
                    ["0", "pothole_confirmed", "duplicate"],
                ]
            ),
            "appears more than once",
        ),
        (
            _review_csv([["0", "Pothole_confirmed", "wrong case"]]),
            "review_status must be exactly",
        ),
        (b"frame_index,review_status,note\n\xff", "not valid UTF-8"),
        (b"frame_index,review_status,note\n\"0,pothole_confirmed,broken", "formatting error"),
    ],
)
def test_frame_review_csv_is_strict(tmp_path, review_bytes, message):
    repo_root = _repo(tmp_path)
    _, errors = prepare_manual_curation_batch(
        _make_annotation_kit(),
        _manual_csv([["POT-001", "0", "10", "20", "50", "60", "pothole", ""]]),
        review_bytes,
        "strict-review",
        output_dir="data/interim/manual_curation/strict-review",
        write=False,
        repo_root=repo_root,
    )
    assert errors
    assert message in errors[0]


def test_unsafe_or_invalid_annotation_kit_is_rejected_without_output(tmp_path):
    repo_root = _repo(tmp_path)
    valid_annotations = _manual_csv([["POT-001", "0", "10", "20", "50", "60", "pothole", ""]])
    reviews = _review_csv([["0", "pothole_confirmed", "reviewed"]])

    unsafe_kit = _make_annotation_kit(extra_members={"../outside.txt": b"bad"})
    _, errors = prepare_manual_curation_batch(
        unsafe_kit,
        valid_annotations,
        reviews,
        "unsafe-kit",
        output_dir="data/interim/manual_curation/unsafe-kit",
        write=False,
        repo_root=repo_root,
    )
    assert errors
    assert "unsafe ZIP member" in errors[0]

    mismatched_kit = _make_annotation_kit(manifest_dimensions=(101, 80))
    _, errors = prepare_manual_curation_batch(
        mismatched_kit,
        valid_annotations,
        reviews,
        "dimension-mismatch",
        output_dir="data/interim/manual_curation/dimension-mismatch",
        write=False,
        repo_root=repo_root,
    )
    assert errors
    assert "dimensions do not match" in errors[0]

    oversized_manifest_kit = _make_annotation_kit(manifest_dimensions=(9000, 80))
    _, errors = prepare_manual_curation_batch(
        oversized_manifest_kit,
        valid_annotations,
        reviews,
        "oversized-image",
        output_dir="data/interim/manual_curation/oversized-image",
        write=False,
        repo_root=repo_root,
    )
    assert errors
    assert "safe image-dimension limit" in errors[0]
    assert not (repo_root / "data").exists()


def test_output_is_confined_and_dry_run_catches_existing_target_conflict(tmp_path):
    repo_root = _repo(tmp_path)
    kit, annotations, reviews = _valid_inputs()

    _, errors = prepare_manual_curation_batch(
        kit,
        annotations,
        reviews,
        "safe-id",
        output_dir="data/interim/manual_curation/another-id",
        write=False,
        repo_root=repo_root,
    )
    assert errors == ["Curation output must be exactly data/interim/manual_curation/<recording_id>."]

    _, errors = prepare_manual_curation_batch(
        kit,
        annotations,
        reviews,
        "../unsafe",
        output_dir="data/interim/manual_curation/unsafe",
        write=False,
        repo_root=repo_root,
    )
    assert errors
    assert "recording_id" in errors[0]

    existing = repo_root / "data/interim/manual_curation/rural-road-session-001"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("must stay")
    _, errors = _run_valid_batch(repo_root, write=False)
    assert errors
    assert "already exists" in errors[0]
    assert sentinel.read_text() == "must stay"


def test_overwrite_is_atomic_and_preserves_old_batch_if_staging_fails(tmp_path, monkeypatch):
    repo_root = _repo(tmp_path)
    summary, errors = _run_valid_batch(repo_root, write=True)
    assert errors == []
    assert summary["dry_run"] is False
    target = repo_root / "data/interim/manual_curation/rural-road-session-001"
    old_manifest = (target / "manifests/curation_manifest.csv").read_bytes()

    def fail_before_promotion(*_args, **_kwargs):
        raise ManualCurationError("synthetic staging failure")

    monkeypatch.setattr(curation, "_write_batch_contents_at", fail_before_promotion)
    _, errors = _run_valid_batch(repo_root, write=True, overwrite=True)
    assert errors == ["synthetic staging failure"]
    assert (target / "manifests/curation_manifest.csv").read_bytes() == old_manifest
    assert not list(target.parent.glob(".rural-road-session-001.stage-*"))
    assert not list(target.parent.glob(".rural-road-session-001.backup-*"))


def test_symlinked_curation_parent_is_rejected(tmp_path):
    repo_root = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo_root / "data").symlink_to(outside, target_is_directory=True)

    _, errors = _run_valid_batch(repo_root, write=False)
    assert errors
    assert "preflight failed" in errors[0] or "unsafe" in errors[0]
    assert not list(outside.iterdir())


def test_curation_does_not_modify_frozen_dataset_or_training_artifacts(tmp_path):
    repo_root = _repo(tmp_path)
    frozen_dataset = repo_root / "data/processed/rdd2022_india_roboflow_d40_v1"
    training_run = repo_root / "outputs/training/frozen-run"
    frozen_dataset.mkdir(parents=True)
    training_run.mkdir(parents=True)
    dataset_sentinel = frozen_dataset / "sentinel.txt"
    training_sentinel = training_run / "sentinel.txt"
    dataset_sentinel.write_text("dataset must not change")
    training_sentinel.write_text("training must not change")
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (dataset_sentinel, training_sentinel)
    }

    _, errors = _run_valid_batch(repo_root, write=True)
    assert errors == []
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert after == before

    target = repo_root / "data/interim/manual_curation/rural-road-session-001"
    produced_names = {path.name for path in target.rglob("*")}
    assert "pothole.yaml" not in produced_names
    assert not any(name.endswith(".pt") for name in produced_names)
    source = Path(curation.__file__).read_text(encoding="utf-8")
    assert "import ultralytics" not in source
    assert "from ultralytics" not in source
    assert "train_pothole" not in source
    assert "evaluate_pothole" not in source


def _load_curation_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "curate_manual_pothole_batch.py"
    spec = importlib.util.spec_from_file_location("curate_manual_pothole_batch_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_dry_run_is_write_free_and_overwrite_requires_write(tmp_path, monkeypatch, capsys):
    module = _load_curation_script_module()
    repo_root = _repo(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    kit, annotations, reviews = _valid_inputs()
    kit_path = tmp_path / "kit.zip"
    annotations_path = tmp_path / "annotations.csv"
    reviews_path = tmp_path / "review.csv"
    kit_path.write_bytes(kit)
    annotations_path.write_bytes(annotations)
    reviews_path.write_bytes(reviews)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "curate_manual_pothole_batch.py",
            "--annotation-kit",
            str(kit_path),
            "--annotations",
            str(annotations_path),
            "--frame-review",
            str(reviews_path),
            "--recording-id",
            "cli-dry-run",
            "--output-dir",
            "data/interim/manual_curation/cli-dry-run",
            "--dry-run",
        ],
    )
    module.main()
    assert "No files were written" in capsys.readouterr().out
    assert not (repo_root / "data").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "curate_manual_pothole_batch.py",
            "--annotation-kit",
            str(kit_path),
            "--annotations",
            str(annotations_path),
            "--frame-review",
            str(reviews_path),
            "--recording-id",
            "cli-dry-run",
            "--output-dir",
            "data/interim/manual_curation/cli-dry-run",
            "--dry-run",
            "--overwrite",
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert "--overwrite is valid only together with --write" in capsys.readouterr().err
