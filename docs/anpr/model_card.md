# RoadSense ANPR model card

## Current status and intended use

RoadSense provides local, operator-assisted inspection of still vehicle images:
plate-region detection, candidate text recognition, a limited `IN_COMMON_V1`
format check, crop-quality cues, and explicit annotated-image/review-JSON exports.
The selected RapidOCR pipeline is **review-only**. Its readings are candidates;
a supported-format match does not verify registration or identity. The current
workflow and its problem-statement coverage are documented in the
[demonstration and jury guide](demo_and_jury_guide.md).

This card summarizes recorded evidence through commit `21586e9`, including the
applied schema 1.2 `candidate_cleanup_v1` policy. It documents existing behavior
without changing model weights, original normalization or acceptance thresholds.
Validation results measure agreement with supplied `publisher_mixed` labels, not an
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
Sources: [detector report](witvix_detector_baseline_results.md#validation-split-val-detection-metrics),
[original OCR comparison](ocr_model_comparison_results.md), and
[applied candidate cleanup results](text_cleanup_policy_results.md).

| Measurement | Recorded result | Interpretation |
| --- | --- | --- |
| Best detector checkpoint, fused validation | Precision 0.998; recall 0.947; mAP50 0.992; mAP50–95 0.847 | Plate-box metrics from Ultralytics, not plate-reading accuracy |
| RapidOCR original normalized-text agreement / CER | 37/94 (39.36%) / 220/914 edits (24.07%) | Preserved pre-cleanup baseline; these fields retain their original meaning |
| RapidOCR with applied candidate policy: exact agreement | **58/94 (61.7%)** | Reviewed `candidate_text` when present, otherwise the unchanged normalized reading; validation-derived |
| Applied candidate-policy character error rate | **168/914 edits (18.38%)** | Offline reassessment of the same frozen OCR reads, not improved OCR inference |
| Cleanup effect on affected plates | **21 fixed / 0 broken** across 23 cleaned plates; 2 remain wrong | Observed on this validation set, not a guarantee for new images |
| Adapted EasyOCR exact agreement / CER | 22/94 (23.40%) / 30.09% | Preserved comparison; not the selected default |

The selected RapidOCR configuration has zero automatically accepted readings
by design. This does not imply zero recognition errors. Its score scale is not
treated as a calibrated probability or substituted into the EasyOCR acceptance
threshold. Cleaned candidates additionally carry `TEXT_CLEANED` and can never
be automatically accepted, regardless of engine or score. See
[review-only behavior](demo_and_jury_guide.md#results-to-present-honestly) and the
[candidate cleanup contract](text_cleanup_policy_results.md#policy).

### Applied candidate cleanup, separate from raw evidence

Schema 1.2 preserves `raw_text`, `normalized_text` and the original
`format_status`. The policy removes only the specified dot-like marks
(`. , : · • °`) and removes a leading `IND` token only if the remainder matches
`IN_COMMON_V1`. It does not substitute look-alike letters/digits. Changes appear
in separate `candidate_text`, `candidate_format_status` and `text_cleanup`
fields, and the UI/export identifies cleaned text. The implementation and
measurement are recorded in [candidate cleanup results](text_cleanup_policy_results.md).

**The policy is now applied; the result is still validation-derived.** It was
designed from these same validation failures and assessed by rescoring frozen
predictions without new OCR inference. **58/94 (61.7%) and 18.38% CER are observed
candidate-policy agreement on supplied labels, not independent accuracy or
passenger-car performance.** The held-out ANPR test evaluation remains pending.
All readings still require human review.

## Known failure areas

The [frozen-prediction failure analysis](ocr_failure_analysis.md) describes the
**original normalized-text baseline**, before candidate cleanup:

- **Extra characters:** 28 of the 57 non-exact readings contain the reference
  characters in order plus extra characters. The analysis identifies leading
  `IND` emblem text and punctuation/fastener or border marks as prominent
  sources. This is the largest measured failure category, not evidence that
  every error can be fixed by cleanup.
- **Low light:** original normalized-text agreement is 7/29; the applied
  candidate policy raises it to 12/29 on the same reads. Genuine mixed character
  errors remain; text cleanup alone does not resolve poor capture. Source:
  [cleanup measurement](text_cleanup_policy_results.md#measurement).
- **Two-line layouts:** the original baseline has 22/63 exact agreements; this
  is not a post-cleanup subgroup result. The analysis reports limited
  pure reordering/fragment cases, so it does not justify assuming row splitting
  will solve the remaining errors.
- **Missed plates:** 3/94 references lack a matching detection. OCR cannot read
  a crop the detector did not produce.

The earlier analysis's cleanup counterfactual is historical. The narrower,
explicit reviewed-candidate policy has since been implemented and measured in
[candidate cleanup results](text_cleanup_policy_results.md). That newer report
supersedes the earlier "not applied" status, while retaining its requirement
for independent confirmation after the final pipeline/settings freeze.

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
