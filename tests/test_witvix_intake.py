"""Focused tests for Witvix ANPR dataset intake and CLI provenance guards."""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.import_witvix_anpr import main as import_witvix_main
from src.anpr.dataset import DatasetValidationError, load_manifest
from src.anpr.witvix_intake import build_witvix_records, write_witvix_manifest
from src.anpr import witvix_intake


def _make_jpeg_bytes(width: int, height: int, color: tuple[int, int, int] = (60, 100, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_manifest_creation_does_not_overwrite_file_created_after_preflight(tmp_path, monkeypatch):
    target = tmp_path / "manifest.jsonl"
    target.write_bytes(b"existing manifest\n")
    # Simulate a competing writer winning after both path preflight checks.
    monkeypatch.setattr(witvix_intake, "_safe_interim_manifest_path", lambda *a, **k: target)
    with pytest.raises(DatasetValidationError, match="already exists"):
        write_witvix_manifest([], target, repo_root=tmp_path)
    assert target.read_bytes() == b"existing manifest\n"


def _create_synthetic_witvix_source(
    tmp_path: Path,
    *,
    plate_texts: tuple[str, str, str] = ("RJ11GB1604", "RJ11GB1604", "24BH1234AA"),
) -> Path:
    """Create a tiny synthetic 2-image, 3-plate Witvix directory matching the real schema."""
    src = tmp_path / "witvix_src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    (src / "plates").mkdir(parents=True)

    # Image 1: night, 200x100, 2 plates
    (src / "images" / "img_0001.jpg").write_bytes(_make_jpeg_bytes(200, 100))
    # Image 2: day, 160x120, 1 plate (outside IN_COMMON_V1 shape to test preservation)
    (src / "images" / "img_0002.jpg").write_bytes(_make_jpeg_bytes(160, 120))

    # Crops
    (src / "plates" / "img_0001_0.jpg").write_bytes(_make_jpeg_bytes(40, 20))
    (src / "plates" / "img_0001_1.jpg").write_bytes(_make_jpeg_bytes(50, 20))
    (src / "plates" / "img_0002_0.jpg").write_bytes(_make_jpeg_bytes(60, 24))

    # YOLO labels:
    # img_0001 plate 0: x=[20, 60], y=[30, 50] on 200x100 -> cx=0.20, cy=0.40, w=0.20, h=0.20
    # img_0001 plate 1: x=[100, 150], y=[60, 80] on 200x100 -> cx=0.625, cy=0.70, w=0.25, h=0.20
    (src / "labels" / "img_0001.txt").write_text(
        "0 0.200000 0.400000 0.200000 0.200000\n"
        "0 0.625000 0.700000 0.250000 0.200000\n",
        encoding="utf-8",
    )
    # img_0002 plate 0: x=[16, 76], y=[24, 48] on 160x120 -> cx=46/160=0.2875, cy=36/120=0.30, w=60/160=0.375, h=24/120=0.20
    (src / "labels" / "img_0002.txt").write_text(
        "0 0.287500 0.300000 0.375000 0.200000\n",
        encoding="utf-8",
    )

    (src / "metadata.csv").write_text(
        "image,width,height,n_plates,state_codes,light\n"
        "img_0001.jpg,200,100,2,RJ,night\n"
        "img_0002.jpg,160,120,1,BH,day\n",
        encoding="utf-8",
    )

    t0, t1, t2 = plate_texts
    (src / "plates.csv").write_text(
        "image,plate_index,crop,x_min,y_min,x_max,y_max,plate_text,state_code,layout\n"
        f"img_0001.jpg,0,img_0001_0.jpg,20,30,60,50,{t0},RJ,two_line\n"
        f"img_0001.jpg,1,img_0001_1.jpg,100,60,150,80,{t1},RJ,single_line\n"
        f"img_0002.jpg,0,img_0002_0.jpg,16,24,76,48,{t2},BH,single_line\n",
        encoding="utf-8",
    )

    coco_payload = {
        "info": {"description": "Synthetic Witvix Fixture", "version": "1.0"},
        "licenses": [{"id": 1, "name": "CC-BY-NC-4.0"}],
        "categories": [{"id": 1, "name": "license_plate", "supercategory": "vehicle"}],
        "images": [
            {"id": 1, "file_name": "images/img_0001.jpg", "width": 200, "height": 100, "license": 1},
            {"id": 2, "file_name": "images/img_0002.jpg", "width": 160, "height": 120, "license": 1},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20.0, 30.0, 40.0, 20.0],
                "area": 800.0,
                "iscrowd": 0,
                "attributes": {"plate_text": t0, "state_code": "RJ", "layout": "two_line"},
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 60.0, 50.0, 20.0],
                "area": 1000.0,
                "iscrowd": 0,
                "attributes": {"plate_text": t1, "state_code": "RJ", "layout": "single_line"},
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [16.0, 24.0, 60.0, 24.0],
                "area": 1440.0,
                "iscrowd": 0,
                "attributes": {"plate_text": t2, "state_code": "BH", "layout": "single_line"},
            },
        ],
    }
    (src / "annotations_coco.json").write_text(json.dumps(coco_payload), encoding="utf-8")
    (src / "data.yaml").write_text("path: .\ntrain: images\nval: images\n", encoding="utf-8")
    (src / "README.md").write_text("# Synthetic Witvix README\n", encoding="utf-8")
    return src


def test_build_witvix_records_maps_provenance_conditions_and_redacts_plate_texts_from_report(
    tmp_path,
):
    plate_strings = ("RJ11GB1604", "MH12AB9876", "24BH1234AA")
    src = _create_synthetic_witvix_source(tmp_path, plate_texts=plate_strings)

    records, report = build_witvix_records(src)

    assert len(records) == 2
    rec1, rec2 = records[0], records[1]

    assert rec1.image_id == "witvix-img_0001"
    assert rec1.image_path == "images/img_0001.jpg"
    assert rec1.group_id == "witvix_vehicle_proxy_img_0001"
    assert rec1.grouping_basis == "publisher_vehicle_proxy"
    assert rec1.source_id == "witvix_original_img_0001"
    assert rec1.conditions == ["low_light"]
    assert len(rec1.plates) == 2
    assert all(p.label_provenance == "publisher_mixed" for p in rec1.plates)
    assert [p.text for p in rec1.plates] == ["RJ11GB1604", "MH12AB9876"]

    assert rec2.image_id == "witvix-img_0002"
    assert rec2.image_path == "images/img_0002.jpg"
    assert rec2.group_id == "witvix_vehicle_proxy_img_0002"
    assert rec2.grouping_basis == "publisher_vehicle_proxy"
    assert rec2.source_id == "witvix_original_img_0002"
    assert rec2.conditions == ["other"]
    assert len(rec2.plates) == 1
    assert rec2.plates[0].text == "24BH1234AA"
    assert rec2.plates[0].label_provenance == "publisher_mixed"

    # Report counts & redaction check
    assert report["counts"]["images"] == 2
    assert report["counts"]["plate_annotations"] == 3
    assert report["counts"]["in_common_v1_format_counts"]["MATCHED"] == 2
    assert report["counts"]["in_common_v1_format_counts"]["NO_MATCH"] == 1

    serialized_report = json.dumps(report)
    for secret_plate in plate_strings:
        assert secret_plate not in serialized_report


def test_build_witvix_records_rejects_csv_coco_yolo_disagreement(tmp_path):
    src = _create_synthetic_witvix_source(tmp_path)
    # Corrupt YOLO label on img_0002
    (src / "labels" / "img_0002.txt").write_text(
        "0 0.500000 0.500000 0.375000 0.200000\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="discrepancy"):
        build_witvix_records(src)


def test_build_witvix_records_rejects_symlinks_and_missing_files(tmp_path):
    src = _create_synthetic_witvix_source(tmp_path)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_make_jpeg_bytes(200, 100))

    (src / "images" / "img_0001.jpg").unlink()
    (src / "images" / "img_0001.jpg").symlink_to(outside)

    with pytest.raises(DatasetValidationError, match="Symlinked"):
        build_witvix_records(src)


def test_cli_dry_run_and_execute_require_acknowledgement_flags_and_confine_output(
    tmp_path, capsys
):
    src = _create_synthetic_witvix_source(tmp_path)
    fake_repo = tmp_path / "repo"
    (fake_repo / "data" / "interim").mkdir(parents=True)

    # 1. Dry-run succeeds without writing any file
    rc = import_witvix_main(
        [
            "--source-root",
            str(src),
            "--output-manifest",
            "data/interim/witvix_manifest.jsonl",
            "--dry-run",
        ],
        repo_root=fake_repo,
    )
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "RJ11GB1604" not in out_text
    assert not (fake_repo / "data" / "interim" / "witvix_manifest.jsonl").exists()

    # 2. Execute without both acknowledgement flags fails closed
    rc_missing_flags = import_witvix_main(
        [
            "--source-root",
            str(src),
            "--output-manifest",
            "data/interim/witvix_manifest.jsonl",
            "--execute",
            "--accept-publisher-labels",
        ],
        repo_root=fake_repo,
    )
    assert rc_missing_flags == 1
    capsys.readouterr()
    assert not (fake_repo / "data" / "interim" / "witvix_manifest.jsonl").exists()

    # 3. Execute outside data/interim/ is rejected
    rc_bad_path = import_witvix_main(
        [
            "--source-root",
            str(src),
            "--output-manifest",
            "data/processed/witvix_manifest.jsonl",
            "--execute",
            "--accept-publisher-labels",
            "--accept-vehicle-proxy-groups",
        ],
        repo_root=fake_repo,
    )
    assert rc_bad_path == 1
    capsys.readouterr()

    # 4. Execute with both flags writes valid manifest that load_manifest accepts
    rc_exec = import_witvix_main(
        [
            "--source-root",
            str(src),
            "--output-manifest",
            "data/interim/witvix_manifest.jsonl",
            "--execute",
            "--accept-publisher-labels",
            "--accept-vehicle-proxy-groups",
        ],
        repo_root=fake_repo,
    )
    assert rc_exec == 0
    exec_out = json.loads(capsys.readouterr().out)
    assert exec_out["mode"] == "execute"
    manifest_file = fake_repo / "data" / "interim" / "witvix_manifest.jsonl"
    assert manifest_file.is_file()

    loaded = load_manifest(manifest_file, prepared=False)
    assert len(loaded) == 2
    assert loaded[0].grouping_basis == "publisher_vehicle_proxy"
    assert loaded[0].plates[0].label_provenance == "publisher_mixed"

    # 5. Second execute targeting existing output file is rejected
    rc_overwrite = import_witvix_main(
        [
            "--source-root",
            str(src),
            "--output-manifest",
            "data/interim/witvix_manifest.jsonl",
            "--execute",
            "--accept-publisher-labels",
            "--accept-vehicle-proxy-groups",
        ],
        repo_root=fake_repo,
    )
    assert rc_overwrite == 1
