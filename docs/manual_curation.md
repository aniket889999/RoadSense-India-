# Manual Curation Pool for Future Experiments

This local-only workflow turns **human-confirmed** sampled frames from a
RoadSense annotation kit into a separate curation pool for a possible future
model experiment. It does not train a model, run inference, change the frozen
RDD2022 dataset, or re-use the frozen baseline's held-out test split.

## Why this exists

The current frozen pothole baseline can miss real potholes on roads that look
different from its training data. Improving a future model requires examples
that a person has reviewed and labelled. Model suggestions are not labels.

The curation tool copies only reviewed sampled JPEG frames from an annotation
kit. It never copies the original MP4 video into the curation pool.

## Inputs

You need three files:

1. `annotation_kit.zip` downloaded from RoadSense.
2. A completed manual annotations CSV with the existing exact header:

   ```csv
   incident_id,frame_index,x_min,y_min,x_max,y_max,label,note
   ```

3. A separate frame-review CSV with this exact header:

   ```csv
   frame_index,review_status,note
   ```

Allowed `review_status` values are:

- `pothole_confirmed`: the frame must contain at least one manually entered
  pothole box.
- `no_pothole_confirmed`: the frame must contain no manually entered box.

Frames that are not listed in the frame-review CSV are excluded. They are
never silently treated as negative training examples.

Example:

```csv
frame_index,review_status,note
0,pothole_confirmed,Visible pothole manually boxed
55,no_pothole_confirmed,Reviewed road surface; no pothole visible
```

## Create a batch

Use one `recording_id` per original road recording. It is a grouping boundary:
all frames from the same recording must stay together in a future split to
reduce leakage. Do not use personal names, GPS coordinates, or a local path as
the ID.

First run a dry run. It validates every input and writes nothing:

```bash
.venv/bin/python scripts/curate_manual_pothole_batch.py \
  --annotation-kit "/path/to/annotation_kit.zip" \
  --annotations "/path/to/completed_manual_annotations.csv" \
  --frame-review "/path/to/confirmed_frame_reviews.csv" \
  --recording-id rural-road-session-001 \
  --output-dir data/interim/manual_curation/rural-road-session-001 \
  --dry-run
```

After reviewing the dry-run summary, explicitly create the local batch:

```bash
.venv/bin/python scripts/curate_manual_pothole_batch.py \
  --annotation-kit "/path/to/annotation_kit.zip" \
  --annotations "/path/to/completed_manual_annotations.csv" \
  --frame-review "/path/to/confirmed_frame_reviews.csv" \
  --recording-id rural-road-session-001 \
  --output-dir data/interim/manual_curation/rural-road-session-001 \
  --write
```

The tool refuses arbitrary output locations. The destination must be exactly:

```text
data/interim/manual_curation/<recording-id>/
```

It also refuses to overwrite a batch unless `--write --overwrite` is supplied.

## Output

The local, Git-ignored batch contains only selected JPEGs, class-0 YOLO labels,
and portable manifests:

```text
data/interim/manual_curation/rural-road-session-001/
├── images/
│   ├── frame_00000.jpg
│   └── frame_00055.jpg
├── labels/
│   ├── frame_00000.txt
│   └── frame_00055.txt
└── manifests/
    ├── curation_metadata.json
    └── curation_manifest.csv
```

To reduce privacy risk, each exported JPEG is freshly encoded without source
EXIF, XMP, comments, or other embedded image metadata. Review and annotation
notes are used for input validation but are not exported. The output therefore
does not contain the original video, absolute local paths, model scores, model
predictions, source-image metadata, the frozen dataset fingerprint, or any
held-out-test result.

## Strict limits

- This is a curation pool, not a ready-to-train dataset.
- It is not merged into `data/processed/rdd2022_india_roboflow_d40_v1`.
- Do not create a new train/validation/test split from one recording alone.
- A future experiment should use multiple independent recording groups and
  define a newly held-out protocol before training or reporting metrics.
- The frozen baseline's one-time test evaluation remains untouched.
