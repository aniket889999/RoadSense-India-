# Passenger-car evaluation set — collection and labelling protocol

The Witvix data is mostly commercial trucks with publisher labels, so it cannot show how
the system performs on ordinary cars, real blur or real camera angles. This protocol
creates a small, **independent, human-labelled** car set to answer that honestly.

## 1. Rules that keep the result honest

- **Independent test set.** Collect and label it, then evaluate the frozen pipeline
  **once**. Do not change thresholds, cleanup rules or models after seeing its results; if
  you do, the set becomes a development set and you must say so.
- **Two independent labels per image.** Two different people transcribe each plate from the
  image without seeing the system's output or each other's answer. Disagreements are
  resolved by a third look (adjudication) and recorded. AI-assisted transcription is not a
  human label.
- **Permission.** Use only your own vehicles, vehicles whose owners agreed, or images you
  are permitted to use. Record which in the `permission` column. Do not scrape images.
- **Privacy.** Plate numbers are personal data (India's DPDP Act 2023). Keep images and
  the manifest in `data/raw/car_eval_v1/` (Git-ignored). Never commit them; report only
  aggregate numbers.

## 2. What to collect

Aim for **at least 100 images** (more is better: with 100 images and 0 errors the 95% upper
bound on the error rate is still about 3%). Spread them across:

| Tag | Values | Target |
| --- | --- | --- |
| `blur` | `none`, `mild` (soft edges, still readable), `strong` (hard to read by eye) | ≥ 20 mild, ≥ 15 strong |
| `angle` | `frontal`, `oblique_lt30` (< 30°), `oblique_ge30` (≥ 30°) | ≥ 20 per oblique bin |
| `lighting` | `day`, `night`, `glare`, `shade` | ≥ 20 night |
| `plate_layout` | `single_line`, `two_line` | as found |
| `vehicle_type` | `car`, `suv`, `other` | mostly `car` / `suv` |

Phone photos at normal parking distances (2–6 m) are fine. Include some deliberately hard
images; the point is to measure failure, not to look good.

## 3. Steps

```bash
mkdir -p data/raw/car_eval_v1/images
.venv/bin/python scripts/car_eval.py template --output data/raw/car_eval_v1/manifest.csv
# copy photos into data/raw/car_eval_v1/images/ and fill one manifest row per image
.venv/bin/python scripts/car_eval.py validate --manifest data/raw/car_eval_v1/manifest.csv \
  --images data/raw/car_eval_v1/images
```

`validate` checks permission values, two different labelers, required adjudication when
labels disagree, tag vocabularies, file names and duplicates. It reports inter-labeler
agreement, which itself is a useful number: it shows how hard the images are for humans.

Only after the pipeline is frozen:

```bash
.venv/bin/python scripts/car_eval.py evaluate --manifest data/raw/car_eval_v1/manifest.csv \
  --images data/raw/car_eval_v1/images --frozen-pipeline-confirmed \
  --output outputs/anpr/car_eval_v1_report.json            # add --auto-accept to measure the policy
```

The report (no plate strings) gives exact agreement, missed plates and automatic
acceptances overall and by blur, angle, lighting, layout and vehicle type. A details file
with strings is written next to it under `outputs/` for local review only.

## 4. What it can and cannot show

It shows image-level reading agreement on your cars under the recorded conditions, with
human-checked labels. It does not show nationwide performance, other cameras or every
plate format, and a set of ~100 images gives wide confidence intervals: report counts
and intervals, not just percentages.
