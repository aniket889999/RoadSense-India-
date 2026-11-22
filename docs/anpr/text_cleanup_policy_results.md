# ANPR candidate cleanup policy (`candidate_cleanup_v1`) — validation results

The [failure analysis](ocr_failure_analysis.md) showed that 28 of 57 validation failures
were the right plate plus extra characters: the `IND` country code printed on Indian
high-security plates, and dot-like marks from bolts and borders. This policy turns that
finding into an explicit, auditable post-OCR step.

## Policy

Applied after the unchanged normalization (uppercase ASCII, remove whitespace/hyphens):

1. Remove only the dot-like marks `. , : · • °`. Characters that could be a misread
   letter or digit (for example `!`, `ß`, full-width letters) are never removed.
2. Remove a leading `IND` token only when the remainder matches `IN_COMMON_V1`.

Nothing is ever substituted (`O`/`0` and similar stay as read). The result is stored in a
new field; existing fields keep their meaning.

| Field (ANPR schema 1.2) | Meaning |
| --- | --- |
| `raw_text` | OCR output exactly as returned (unchanged) |
| `normalized_text` | case/whitespace/hyphen normalization (unchanged) |
| `format_status` | format check of `normalized_text` (unchanged meaning) |
| `candidate_text` | present only when a cleanup step removed something |
| `candidate_format_status` | format check of `candidate_text` |
| `text_cleanup` | the steps applied: `PUNCTUATION_REMOVED`, `LEADING_EMBLEM_REMOVED` |

A cleaned read always carries the review reason `TEXT_CLEANED`, so it can never be
automatically accepted, whatever the OCR engine or score. Saved schema 1.0/1.1 results
remain readable. The UI shows the cleaned candidate next to the raw and normalized text,
and the annotated PNG marks it `(cleaned)`.

## Measurement

The frozen RapidOCR validation reads (`witvix_val_predictions_rapidocr_v1.jsonl`, sha256
`cd851c83ed2c…`) were re-assessed offline with `scripts/rescore_anpr_text_policy.py`. OCR
output, boxes and detection scores are identical; only the post-OCR text assessment was
recomputed. No inference was run and no test-split data was used.

| Validation, 94 plates | `normalized_text` (before) | `candidate_text` (policy) |
| --- | ---: | ---: |
| Exact full-plate agreement | 37 (39.4%) | **58 (61.7%)** |
| Character errors / CER | 220 / 24.07% | **168 / 18.38%** |
| Day / other exact | 30 / 65 | 46 / 65 |
| Low-light exact | 7 / 29 | 12 / 29 |
| Automatically accepted reads | 0 | 0 |

Effect on the 23 plates where a cleanup step applied: **21 fixed, 0 broken**, 2 still
wrong. Steps applied: `PUNCTUATION_REMOVED` 13 times, `LEADING_EMBLEM_REMOVED` 11 times.

## How to present this honestly

- The rule was **derived from the validation failures** and measured on the same
  validation reads. 58/94 is therefore an optimistic estimate, not an independent result.
  It becomes a claim only after the frozen pipeline is evaluated once on the held-out
  test split (and the planned car collection).
- Agreement is with `publisher_mixed` labels on mostly commercial-truck images, not
  human-verified accuracy and not passenger-car performance.
- The rule removes text that is not part of a registration; it does not read characters
  better. Low-light misreads (33% CER after cleanup) and missed plates are unchanged.
- Every reading still requires human review.

Reproduce:

```bash
.venv/bin/python scripts/rescore_anpr_text_policy.py \
  --dataset data/processed/witvix_anpr_pilot_v1 \
  --predictions outputs/anpr/witvix_val_predictions_rapidocr_v1.jsonl \
  --output-predictions outputs/anpr/<new>.jsonl --output-report outputs/anpr/<new>.json
```
