# RoadSense ANPR data card

## Source, attribution and permissions

The pilot uses the **Witvix Indian Truck License Plates sample**, attributed to
[Witvix and its published dataset repository](https://github.com/witvix/indian-truck-license-plates).
The supplied source README states **CC BY-NC 4.0** for research and other
non-commercial use. Preserve that attribution and the source licence when using
the data; this project does not infer commercial permission or grant rights to
redistribute third-party images. Upstream model/software licences are separate.
Source for attribution, licence and all dataset figures in this card:
[local intake report](witvix_intake_results.md).

The downloaded source is `/Users/aniket/Downloads/archive`. Preparation produced
`data/processed/witvix_anpr_pilot_v1`; raw images, plate crops, transcriptions,
manifests and prepared data remain excluded from Git. This card contains no
registration strings or source images.

## Inventory and prepared split

The audited source contains **470 full images, 470 YOLO label files, 589 plate
annotations and 589 supplied crops**. The importer cross-checks CSV, COCO and
YOLO geometry and preserves text labels. The supplied showcase `data.yaml`
points train and validation at the same images; the project does not use that
as an evaluation split. See the [inventory and preparation evidence](witvix_intake_results.md).

| Prepared split | Images | Plate annotations | Negative images |
| --- | ---: | ---: | ---: |
| Training | 329 | 408 | 0 |
| Validation | 71 | 94 | 0 |
| Held-out test — untouched | 70 | 87 | 0 |
| Total | 470 | 589 | 0 |

Dataset schema: `1.1`. Prepared dataset SHA-256 fingerprint:

```text
2c33c828642ba484c4698e561df47193975f491291718a44c2d8d853e0c94a4a
```

These counts and identity come from the
[prepared pilot record](witvix_intake_results.md#prepared-pilot-and-readiness-evidence),
not a new inspection of test images. Detector/OCR experiments reported so far
use training and validation data; held-out ANPR test inference and evaluation
remain pending. See the [current demo evidence](demo_and_jury_guide.md#remaining-submission-work).

## Label provenance and grouping limits

- **`publisher_mixed`:** the publisher reports human checking of 341/589
  transcriptions but does not identify the checked annotations. All 589 retain
  mixed provenance; **0 annotations are flagged `human_verified` by this
  project**. This does not assert the publisher performed no human review.
- **`publisher_vehicle_proxy`:** groups derive from image stems and the source's
  one-photo-per-vehicle description. Preparation also joins shared normalized
  text identities, source identities and duplicate raw/prepared image hashes.
  The 470 proxy groups are atomic, with no measured cross-split overlap in those
  identities/hashes. This is not verified capture-session, camera or site
  disjointness; metadata records `not_established_vehicle_proxy`.
- **Text is preserved:** 583 annotations match the limited `IN_COMMON_V1`
  profile and 6 do not. Nonmatching labels are retained, not rewritten to satisfy
  a format rule. Format agreement does not verify registration.

Source: [intake provenance and preparation decisions](witvix_intake_results.md#evidence-limitations-and-preparation-decisions).
Reported OCR results must therefore be called **agreement with supplied labels**;
independently human-verified reference text would be needed for that stronger
evaluation claim.

## Representation and appropriate interpretation

The sample mainly represents commercial trucks and excludes plates the source
authors could not read. It contains 270 day images tagged `other` and 200 night
images tagged `low_light`; daylight is not evidence of a sharp or frontal view.
It has no true-negative images, so false-positive behavior on plate-free scenes
is unmeasured. Source: [intake limitations](witvix_intake_results.md).

The pilot does not establish ordinary passenger-car performance, unreadable-plate
coverage, capture-session independence or general Indian ANPR reliability.
The [model card](model_card.md) separates detector metrics from OCR agreement
and records measured failure areas. Future permitted car/blur/angle examples
need independent labels and a documented collection protocol; they are not
already part of this evidence. The reserved test split must remain unseen until
the authorized final model/policy freeze and evaluation.

Images and text labels can be sensitive even when distributed as research data.
See [privacy and retention](privacy_and_retention.md); a Git exclusion is not an
access-control or anonymization mechanism.
