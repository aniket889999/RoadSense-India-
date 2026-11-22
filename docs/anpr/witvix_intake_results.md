# Witvix pilot: local dataset intake

Source: `/Users/aniket/Downloads/archive` (read-only).
Publisher: [Witvix Indian Truck License Plates sample](https://github.com/witvix/indian-truck-license-plates).
Licence stated in the supplied README: CC BY-NC 4.0, with attribution for research
and other non-commercial uses. The raw dataset and prepared images stay outside
Git under the existing data exclusions.

## Observed source inventory

| Measurement | Result |
| --- | ---: |
| Full images / YOLO label files | 470 / 470 |
| Plate annotations / plate crops | 589 / 589 |
| Day / night images | 270 / 200 |
| Two-line / single-line plate annotations | 404 / 185 |
| Images with more than one plate annotation | 118 |
| Unique normalized plate strings | 470 |
| Plate strings shared across different source images | 0 |
| Exact duplicate full-image groups | 0 |
| Missing or undecodable full images | 0 |
| Missing referenced labels/crops | 0 |
| Invalid or out-of-bounds CSV plate boxes | 0 |
| CSV / YOLO / COCO geometry inconsistencies | 0 |
| Current image-limit violations | 0 |
| IN_COMMON_V1 matches / nonmatches | 583 / 6 |

Geometry comparisons allowed up to 1.1 pixels for the rounded YOLO/COCO exports.
CSV plate counts agreed with image metadata and label-file counts. Full JPEG
decoding and dimensions were checked, not just filename pairing. Maximum observed
image size was 2,433,600 pixels. No nontrivial EXIF orientation was observed.

The source inventory digest is SHA-256 over sorted relative filenames plus their
individual file SHA-256 values:

```text
ce6f989746e505c46e91eb9adf6e425d317679b731c643724eb20492f3d76a30
```

## Evidence limitations and preparation decisions

The supplied `data.yaml` points both `train` and `val` to `images`. It is a
showcase configuration, not an independent evaluation split, and is not used for
training this project.

The publisher describes one photo per vehicle and prior near-duplicate removal.
The provided metadata does not identify capture sessions/sites. Group IDs derived
from image stems are therefore explicitly `publisher_vehicle_proxy`, not verified
capture-session identities. Existing grouping still joins shared text identities,
source IDs and duplicate raw/normalized image hashes. This cannot establish
independence of backgrounds or capture sessions that the source does not expose.

The supplied README says 341 of 589 plate labels were human checked, but the files
do not identify which ones. Every imported annotation is marked
`label_provenance: publisher_mixed`; none is silently upgraded to human verified.
All six strings outside the current format profile remain unchanged.

Night images are tagged `low_light`; day images are tagged `other`, because
daylight alone does not establish that an image is unblurred or frontal. The
sample excludes plates its authors could not read and mainly represents commercial
trucks, so a pilot result does not establish private-car or unreadable-plate
performance.

Dataset schema 1.1 preserves these grouping and annotation assumptions in the
fingerprint, prepared metadata and training provenance. Offline evaluation of
mixed labels is described as agreement with supplied labels. Human-verified
reference labels are required for a corresponding human-verified accuracy claim.

## Source manifest

The audited CSV was mapped to
`data/interim/witvix_anpr_pilot_v1/manifest.jsonl` with 470 records / 589
annotations. The manifest uses the new provenance fields and references original
images under the read-only source directory. It does not adopt the showcase split,
change plate text or copy the separately supplied plate crops into split folders.

Preparation uses the existing shared dataset tool:

```bash
.venv/bin/python scripts/prepare_anpr_dataset.py \
  --manifest data/interim/witvix_anpr_pilot_v1/manifest.jsonl \
  --source-root /Users/aniket/Downloads/archive \
  --output-dir data/processed/witvix_anpr_pilot_v1 \
  --seed 42 --dry-run
```

Materialization is a separate execution with `--execute`, after the dry run.
The resulting dataset is a limited pilot, not a verified general ANPR benchmark.

The reviewed `scripts/import_witvix_anpr.py` reproduces the source records from
CSV/COCO/YOLO. Its output records were compared with this manifest and matched
exactly. The importer writes a new manifest only with both provenance
acknowledgements (`--accept-publisher-labels` and
`--accept-vehicle-proxy-groups`); it does not overwrite existing manifests.
It never prints registration numbers in its aggregate report.

## Prepared pilot and readiness evidence

Preparation with `--execute` completed and validated
`data/processed/witvix_anpr_pilot_v1`. Its fingerprint matches the dry run:

```text
2c33c828642ba484c4698e561df47193975f491291718a44c2d8d853e0c94a4a
```

| Split | Images | Plate annotations | Negative images |
| --- | ---: | ---: | ---: |
| Train | 329 | 408 | 0 |
| Val | 71 | 94 | 0 |
| Test | 70 | 87 | 0 |
| Total | 470 | 589 | 0 |

All 470 proxy groups are atomic. Image IDs, source IDs, declared group IDs,
normalized plate identities and exact raw/prepared image hashes have zero
cross-split overlap. This does not establish capture-session independence.
No true negative images are available in this sample; false-positive behavior
on scenes without plates remains unmeasured.

The prepared dataset has schema 1.1 and records all 589 labels as
`publisher_mixed`, with `all_plate_texts_human_verified: false` and
`capture_session_disjointness: not_established_vehicle_proxy`.

```bash
.venv/bin/python scripts/train_anpr_detector.py \
  --config configs/training/anpr_plate_detector.yaml \
  --dataset data/processed/witvix_anpr_pilot_v1 --dry-run
```

Training preflight passed for the current CPU / 50 epochs / image size 640 /
batch 8 configuration and the pinned local base-weight SHA. It created no
training run and loaded no model. Accelerator availability is checked only on
actual execution. No detector training, OCR inference or held-out evaluation
was performed.

The complete source inventory SHA was recomputed after preparation and matched
the digest above. Original source files remain unchanged. Raw and prepared data
are ignored by Git; only implementation and documentation are committed.

The combined focused ANPR, intake, API and OpenAPI suite passed 102 tests.
Changed Python modules compiled cleanly and `git diff --check` passed.
