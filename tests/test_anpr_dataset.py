"""Grouped dataset preparation tests with explicitly synthetic labelled images."""

import hashlib
import json
import sys

from PIL import Image
import pytest

from src.anpr.dataset import (
    DatasetValidationError,
    load_manifest,
    prepare_dataset,
    validate_prepared_dataset,
)


@pytest.fixture
def source_dataset(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    (repository / "data" / "processed").mkdir(parents=True)
    monkeypatch.setattr("src.anpr.dataset.REPO_ROOT", repository)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index in range(9):
        name = f"image-{index}.png"
        Image.new("RGB", (80, 40), (index * 20, 50, 80)).save(source / name)
        rows.append({
            "image_id": f"image-{index}", "image_path": name,
            "group_id": f"capture-{index}", "source_id": f"source-{index}",
            "plates": [{"bbox": {"x1": 5, "y1": 5, "x2": 70, "y2": 30},
                        "text": f"MH12AB{index:04d}"}],
            "conditions": ["clear"],
        })
    manifest = source / "annotations.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return repository, source, manifest, rows


def write_rows(manifest, rows):
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_dry_run_is_deterministic_and_does_not_write_or_modify_sources(source_dataset):
    repository, source, manifest, rows = source_dataset
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    output = repository / "data" / "processed" / "anpr-v1"
    first = prepare_dataset(manifest, source, output)
    second = prepare_dataset(manifest, source, output)
    assert first == second
    assert sum(count["images"] for count in first["split_counts"].values()) == 9
    assert all(count["plates"] > 0 for count in first["split_counts"].values())
    assert not output.exists()
    assert not list(output.parent.iterdir())
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}


def test_materialized_dataset_has_matching_images_labels_and_fingerprint(source_dataset):
    repository, source, manifest, _ = source_dataset
    output = repository / "data" / "processed" / "anpr-v1"
    dry = prepare_dataset(manifest, source, output)
    prepared = prepare_dataset(manifest, source, output, dry_run=False)
    assert prepared == dry
    assert validate_prepared_dataset(output) == prepared
    records = load_manifest(output / "manifest.jsonl", prepared=True)
    assert len(records) == 9
    assert all((output / record.image_path).is_file() for record in records)
    assert not list(output.parent.glob(".anpr_stage_*"))


def test_capture_source_and_plate_identity_links_never_cross_splits(source_dataset):
    repository, source, manifest, rows = source_dataset
    rows[1]["group_id"] = rows[0]["group_id"]
    rows[3]["source_id"] = rows[2]["source_id"]
    rows[5]["plates"][0]["text"] = rows[4]["plates"][0]["text"]
    write_rows(manifest, rows)
    output = repository / "data" / "processed" / "grouped"
    prepare_dataset(manifest, source, output, dry_run=False)
    split = {r.image_id: r.split for r in load_manifest(output / "manifest.jsonl", prepared=True)}
    assert split["image-0"] == split["image-1"]
    assert split["image-2"] == split["image-3"]
    assert split["image-4"] == split["image-5"]


def test_insufficient_independent_groups_are_rejected(source_dataset):
    repository, source, manifest, rows = source_dataset
    for row in rows:
        row["group_id"] = "one-shared-session"
    write_rows(manifest, rows)
    with pytest.raises(DatasetValidationError, match="three independent groups"):
        prepare_dataset(manifest, source, repository / "data" / "processed" / "unsafe")


def test_identical_bytes_with_conflicting_labels_are_rejected(source_dataset):
    repository, source, manifest, rows = source_dataset
    (source / "image-1.png").write_bytes((source / "image-0.png").read_bytes())
    with pytest.raises(DatasetValidationError, match="conflicting"):
        prepare_dataset(manifest, source, repository / "data" / "processed" / "conflict")


def test_out_of_bounds_ground_truth_is_rejected(source_dataset):
    repository, source, manifest, rows = source_dataset
    rows[0]["plates"][0]["bbox"]["x2"] = 90
    write_rows(manifest, rows)
    with pytest.raises(DatasetValidationError, match="bounds"):
        prepare_dataset(manifest, source, repository / "data" / "processed" / "invalid")


def test_sources_cannot_escape_the_declared_root(source_dataset):
    repository, source, manifest, rows = source_dataset
    rows[0]["image_path"] = "../outside.png"
    write_rows(manifest, rows)
    with pytest.raises(DatasetValidationError, match="traversal"):
        prepare_dataset(manifest, source, repository / "data" / "processed" / "escape")


def test_prepared_image_or_label_tampering_is_detected(source_dataset):
    repository, source, manifest, _ = source_dataset
    output = repository / "data" / "processed" / "anpr-v1"
    prepare_dataset(manifest, source, output, dry_run=False)
    record = load_manifest(output / "manifest.jsonl", prepared=True)[0]
    label = output / "labels" / record.split / f"{record.image_id}.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(DatasetValidationError, match="labels"):
        validate_prepared_dataset(output)


def test_existing_output_cannot_be_overwritten(source_dataset):
    repository, source, manifest, _ = source_dataset
    output = repository / "data" / "processed" / "keep"
    output.mkdir()
    marker = output / "user-file.txt"
    marker.write_text("keep me")
    with pytest.raises(DatasetValidationError, match="already exists"):
        prepare_dataset(manifest, source, output, dry_run=False)
    assert marker.read_text() == "keep me"


def test_scoring_cli_defaults_to_validation_and_preserves_existing_report(source_dataset, monkeypatch):
    import scripts.evaluate_anpr_predictions as cli

    repository, source, manifest, _ = source_dataset
    output = repository / "data" / "processed" / "anpr-v1"
    metadata = prepare_dataset(manifest, source, output, dry_run=False)
    validation_records = [r for r in load_manifest(output / "manifest.jsonl", prepared=True) if r.split == "val"]
    predictions = repository / "frozen-predictions.jsonl"
    predictions.write_text("".join(json.dumps({
        "image_id": r.image_id,
        "result": {"schema_version": "1.0", "request_id": r.image_id, "status": "NO_PLATE",
                   "image_width": r.width, "image_height": r.height, "plates": [],
                   "processing_ms": 10.0, "warnings": []},
    }) + "\n" for r in validation_records))
    report = repository / "outputs" / "validation.json"
    monkeypatch.setattr(cli, "REPO_ROOT", repository)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *args, **kwargs: "c" * 40)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_anpr_predictions.py", "--dataset", str(output),
        "--predictions", str(predictions), "--output", str(report),
    ])
    assert cli.main() == 0
    saved = report.read_bytes()
    result = json.loads(saved)
    assert result["split"] == "val"
    assert result["dataset_fingerprint"] == metadata["dataset_fingerprint"]
    assert result["overall"]["missed_plates"] == len(validation_records)
    assert cli.main() == 1
    assert report.read_bytes() == saved
