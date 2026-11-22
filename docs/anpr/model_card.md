# RoadSense ANPR model card

## Current status and intended use

RoadSense provides local, operator-assisted inspection of still vehicle images:
plate-region detection, candidate text recognition, a limited `IN_COMMON_V1`
format check, crop-quality cues, and explicit annotated-image/review-JSON exports.
The selected RapidOCR pipeline is **review-only**. Its readings are candidates;
a supported-format match does not verify registration or identity. The current
workflow and its problem-statement coverage are documented in the
[demonstration and jury guide](demo_and_jury_guide.md).

This card summarizes recorded evidence through commit `eef6db8`. It introduces
no model, normalization, threshold or acceptance-policy changes. Validation
results measure agreement with supplied `publisher_mixed` labels, not an
independent human-verified accuracy estimate. The ANPR held-out test split has
not been evaluated. See the [data card](data_card.md).

## Components: trained here versus pretrained

| Component | Project contribution and execution | Evidence |
| --- | --- | --- |
| Plate-region detector | Pretrained YOLOv8n base fine-tuned into a single-class `license_plate` detector; 50 epochs on Apple M4 MPS, image size 640, batch 8, seed 42 | [Detector training report](witvix_detector_baseline_results.md) |
| Selected OCR | Pretrained RapidOCR PP-OCRv4 ONNX pipeline, not trained from scratch or fine-tuned here; bounded `resized_bgr` plate crops; OCR runs on CPU while the detector uses MPS in the recorded configuration | [OCR comparison](ocr_model_comparison_results.md), [current launch guide](demo_and_jury_guide.md) |
| Preserved OCR comparison | EasyOCR English Gen2 recognizer adapted locally using single-line training crops; 15 epochs, with model operations on MPS and CTC loss on CPU; checkpoint selection used an internal training-development subset | [Adaptation report](ocr_adaptation_results.md) |
| Capture-quality review | OpenCV detail, illumination and supported line-tilt measurements; advisory heuristics, not a trained readability classifier or a perspective-recovery model | [Quality evidence and limits](demo_and_jury_guide.md#blur-and-tilt-evidence) |

The selected launcher uses pinned local assets and an explicit detector device.
An unavailable configured device or missing/incompatible assets must remain an
unavailable runtime, rather than trigger an implicit CPU fallback or substitute
a pothole/vehicle detector. Preflight and launch instructions are in the
[demo guide](demo_and_jury_guide.md#reproduce-the-local-demo).

The plate detector's installed checkpoint is
`models/anpr/anpr_witvix_detector_mps_v1_best.pt`, with SHA-256
`37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db`.
The digest is recorded in the [detector artifact table](witvix_detector_baseline_results.md#verified-run-artifacts-and-sha-256-hashes);
the [demo guide](demo_and_jury_guide.md) describes the current installed-assets
preflight. Historical provisioning notes in earlier reports describe their
milestone state, not today's launcher configuration.

## Recorded validation results

All results below use the same validation split: **71 images / 94 supplied plate
annotations**. Missed reference plates remain in the end-to-end reading
denominator. These are validation results; no held-out test result is claimed.
Sources: [detector report](witvix_detector_baseline_results.md#validation-split-val-detection-metrics)
and [OCR comparison](ocr_model_comparison_results.md).

| Measurement | Recorded result | Interpretation |
| --- | --- | --- |
| Best detector checkpoint, fused validation | Precision 0.998; recall 0.947; mAP50 0.992; mAP50–95 0.847 | Plate-box metrics from Ultralytics, not plate-reading accuracy |
| Selected RapidOCR exact full-plate agreement | **37/94 (39.36%)** | Complete normalized reading agrees with the supplied reference |
| Selected RapidOCR character error rate | **220/914 character edits (24.07%)** | Character-level error, not a confidence probability |
| Adapted EasyOCR exact agreement / CER | 22/94 (23.40%) / 30.09% | Preserved comparison; not the selected default |

The selected RapidOCR configuration has zero automatically accepted readings
by design. This does not imply zero recognition errors. Its score scale is not
treated as a calibrated probability or substituted into the EasyOCR acceptance
threshold. See [review-only behavior](demo_and_jury_guide.md#results-to-present-honestly).

## Known failure areas

The [frozen-prediction failure analysis](ocr_failure_analysis.md) reproduces the
recorded validation totals without new inference:

- **Extra characters:** 28 of the 57 non-exact readings contain the reference
  characters in order plus extra characters. The analysis identifies leading
  `IND` emblem text and punctuation/fastener or border marks as prominent
  sources. This is the largest measured failure category, not evidence that
  every error can be fixed by cleanup.
- **Low light:** only 7/29 reference plates have exact agreement. Genuine mixed
  character errors remain; text cleanup alone does not resolve poor capture.
- **Two-line layouts:** 22/63 exact agreements. The analysis reports limited
  pure reordering/fragment cases, so it does not justify assuming row splitting
  will solve the remaining errors.
- **Missed plates:** 3/94 references lack a matching detection. OCR cannot read
  a crop the detector did not produce.

**Unapplied diagnostic counterfactual:** the analysis explores a **37→58/94**
change by removing punctuation and a leading `IND` token from the same frozen
validation reads. **That policy is not applied to the pipeline. 58/94 is not
achieved application accuracy, a new frozen-model result, or independent
evaluation evidence.** The achieved selected-pipeline result remains 37/94.
Any future change belongs to a separate raw-evidence-preserving policy,
validation experiment and independent confirmation after settings are frozen.
See the [counterfactual and its restrictions](ocr_failure_analysis.md#2-what-the-extra-characters-are).

Blur, glare and strong perspective remain capture risks. Quality cues suggest
review or recapture; they do not reconstruct missing characters, invent plate
corners, measure vehicle depth, or establish real blur/skew classification
accuracy. See [quality limitations](demo_and_jury_guide.md#blur-and-tilt-evidence).

## Uses outside the evidence boundary

Do not use this prototype to assert registration validity, identify an owner,
associate a plate with a particular parking bay, control a toll/barrier, charge
a vehicle, or make unattended enforcement/safety decisions. These integrations
are not implemented. No measured passenger-car reading accuracy, broad Indian
plate coverage, deployment certification or live-throughput guarantee is
established. The truck-heavy, mixed-label sample and absent negative images
limit generalization; see the [data card](data_card.md) and
[remaining submission work](demo_and_jury_guide.md#remaining-submission-work).

Follow the [privacy and retention card](privacy_and_retention.md) when handling
vehicle images or reading exports. Dataset permissions and upstream component
licences must be checked separately before redistribution or commercial use.
