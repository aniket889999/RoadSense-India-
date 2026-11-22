# ANPR automatic acceptance (`rapidocr_gated_v1`) — validation results

By default every RapidOCR reading goes to human review. `rapidocr_gated_v1` is an
**opt-in** policy that lets the system accept a reading on its own only when several
independent signals agree. It is off unless the API is started with:

```bash
.venv/bin/python scripts/serve_anpr_local.py --auto-accept
```

A reading is accepted (`READ`) only if **all** hold:

1. RapidOCR's own score is ≥ 0.90 (the EasyOCR 0.75 threshold is not reused: the scales differ);
2. the normalized text matches `IN_COMMON_V1` and its state code is a known state/UT code
   (`IN_PLATE_V2`);
3. no candidate cleanup was applied — cleaned readings always stay in review.

Everything else remains `NEEDS_REVIEW` with explicit reasons. The launcher never inherits
the policy from the shell; it must be chosen with the flag.

## How the threshold was chosen (validation only)

`scripts/measure_anpr_acceptance.py` re-ran the same detector, crops and RapidOCR on the 71
validation images with the score retained. Raw OCR text matched the frozen review-only run
on all 71 images. Risk/coverage over 94 reference plates:

| Rule | Accepted | Wrong | 95% upper bound on wrong-acceptance rate |
| --- | ---: | ---: | ---: |
| Score alone, strictest zero-error point (0.994) | 5 | 0 | 45% |
| Format gate + score ≥ 0.8774 (tightest zero-error point) | 51 | 0 | 5.7% |
| Format gate + score ≥ 0.90 | 47 | 0 | 6.2% |
| Format gate, any score | 62 | 4 | 14.2% |

The score alone does not separate right from wrong readings; combined with the format
gate it does. 0.8774 sits immediately above a wrong reading (0.8755), so 0.90 was chosen
to leave a margin. The upper bound is the exact one-sided 95% Clopper–Pearson limit.

## End-to-end result of the implemented policy

The implemented policy was then run end to end on the validation split
(`scripts/generate_anpr_predictions.py` with `--auto-accept` settings; predictions sha256
`1d37332380…`, provenance records `acceptance_policy: rapidocr_gated_v1`, threshold 0.9):

| Validation, 94 plates | Value |
| --- | ---: |
| Automatically accepted readings | **33 (35.1%)** |
| Wrong accepted readings | **0** |
| 95% upper bound on wrong-acceptance rate (0 of 33) | 8.7% |
| Readings left for review | 61 |
| Exact agreement of the review text (candidate) | 58 / 94 (61.7%), CER 18.38% |

The measured coverage (33) is lower than the risk/coverage table's 47 because cleaned
readings are excluded by design. Review reasons on the 61 remaining readings:
low score 34, unsupported format 53, cleaned text 23, unknown state code 1 (a reading can
have several). Mean processing time was 567 ms per image (MPS detector, CPU OCR).

## How to present this

- "With the opt-in policy, about a third of validation plates were accepted automatically
  with no errors; the other two thirds were routed to a reviewer with a reason."
- "Zero errors in 33 does not mean zero risk: the 95% upper bound is about 9%." Say this
  unprompted — it shows you understand small-sample evidence.
- The threshold was chosen on validation. The policy is now frozen; its first independent
  check is the one-time held-out test evaluation (and the planned car photo set).
- Labels are `publisher_mixed` on mostly truck images; accuracy on passenger cars is not
  established. The default demo remains review-only.
