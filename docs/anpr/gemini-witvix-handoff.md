# RoadSense ANPR — Gemini Witvix Dataset Intake Handoff

## Git State

- **Branch**: `feat/anpr-foundation`
- **Starting Checkpoint (HEAD SHA)**: `dbdd0f5`

## Gemini-Owned Intake Files

| File | Purpose |
| --- | --- |
| `src/anpr/witvix_intake.py` | Implements `build_witvix_records(source_root: Path) -> tuple[list[ImageRecord], dict]` and `write_witvix_manifest(...)` with full file confinement, decoded image dimension checks, and CSV / COCO / YOLO agreement validation |
| `scripts/import_witvix_anpr.py` | CLI wrapper supporting `--dry-run` (default) and `--execute` gated by `--accept-publisher-labels` and `--accept-vehicle-proxy-groups` |
| `tests/test_witvix_intake.py` | Focused unit tests with tiny synthetic Witvix fixtures verifying record mapping, registration-number redaction in reports, CSV/COCO/YOLO disagreement rejection, symlink rejection, and CLI execution guards |
| `docs/anpr/gemini-witvix-handoff.md` | This handoff document |

## Contract Mapping & Provenance Rules

Every record returned by `build_witvix_records(source_root)` conforms to `src.anpr.dataset.ImageRecord`:
- `image_id = f"witvix-{image_stem}"`
- `image_path = f"images/{filename}"`
- `group_id = f"witvix_vehicle_proxy_{image_stem}"`
- `grouping_basis = "publisher_vehicle_proxy"`
- `source_id = f"witvix_original_{image_stem}"`
- `conditions = ["low_light"]` when `light == "night"`, or `["other"]` when `light == "day"` (daylight does not prove clear/unblurred imagery).
- Each `PlateAnnotation` sets `label_provenance = "publisher_mixed"` (never guessing which 341 of 589 plates were human checked) and preserves all original `plate_text` strings verbatim (583 matching `IN_COMMON_V1` and 6 outside `IN_COMMON_V1`).
- The supplied `data.yaml` (`train: images`, `val: images`) is ignored for split assignment.
- No registration numbers are printed in the report dictionary or CLI output.

## Read-Only Dry-Run Audit of `/Users/aniket/Downloads/archive`

- **Counts**:
  - `images`: `470` (`352` single-plate, `118` multi-plate)
  - `yolo_label_files`: `470`
  - `plate_annotations`: `589` (`185` `single_line`, `404` `two_line`)
  - `crop_files`: `589`
  - `light_distribution`: `270` `day`, `200` `night`
  - `in_common_v1_format_counts`: `583` `MATCHED`, `6` `NO_MATCH`, `0` `EMPTY`
- **Geometry Agreement**:
  - `csv_coco_yolo_consistent`: `true`
  - `max_csv_coco_diff_px`: `0.5`
  - `max_csv_yolo_diff_px`: `0.501`
  - `max_coco_yolo_diff_px`: `0.001098`
- **Source File SHA-256 Hashes**:
  - `metadata.csv`: `1ad438d45effa6bbee66403bc408e7401a04f1668227223c2d4a702afab125ae`
  - `plates.csv`: `475123d3030ec47dbfc7c3fcee6722fe20a819581bd93f5e0b085f32b9e59bd2`
  - `annotations_coco.json`: `6f9fb261116ca978ae0731c8b33bb768e6310af66fb5aa203c5105df8d20abf1`
  - `data.yaml`: `54bef923bcda945f37964c0b84a85e0fd9df1c3e6bbf2630dc05a53cfa6cb7e8`
  - `README.md`: `592fc280125ed53170d886f9b28a8b025f5e79add17ffa890df1203ace8d3a61`

## CLI Usage

### Dry-Run Audit (Default — No Files Written)
```bash
.venv/bin/python scripts/import_witvix_anpr.py \
  --source-root /Users/aniket/Downloads/archive \
  --output-manifest data/interim/witvix_anpr_manifest.jsonl \
  --dry-run
```

### Execute Mode (Writes New Manifest Under `data/interim/`)
```bash
.venv/bin/python scripts/import_witvix_anpr.py \
  --source-root /Users/aniket/Downloads/archive \
  --output-manifest data/interim/witvix_anpr_manifest.jsonl \
  --execute \
  --accept-publisher-labels \
  --accept-vehicle-proxy-groups
```

## Focused Test Results

- `.venv/bin/pytest -q tests/test_witvix_intake.py`: **4 passed in 0.34s**.

## Unresolved Intake Mismatch Check

- **None**: All 470 images, 470 YOLO label files, 589 crops, `metadata.csv`, `plates.csv`, and `annotations_coco.json` in `/Users/aniket/Downloads/archive` agree on image dimensions, box coordinates, and attributes, and conform to `src.anpr.dataset.ImageRecord` (`DATASET_SCHEMA_VERSION = "1.1"`).
