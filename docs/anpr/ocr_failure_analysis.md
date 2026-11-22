# ANPR OCR failure analysis — RapidOCR on validation

Where does the selected OCR pipeline fail, and which failures are worth fixing first?
This analysis re-reads the **frozen** RapidOCR validation predictions; it runs no new
inference and touches no test-split data.

| Item | Value |
| --- | --- |
| Predictions | `outputs/anpr/witvix_val_predictions_rapidocr_v1.jsonl` (sha256 `cd851c83ed2c…`) |
| Split | validation only — 71 images, 94 reference plates (the script cannot select `test`) |
| Matching | greedy IoU ≥ 0.5, text never used for matching (same as `src/anpr/evaluation.py`) |
| Cross-check | totals reproduce the frozen report exactly: 37 exact reads, 220 character errors |
| Labels | `publisher_mixed`, not human-verified → read every number as *agreement with supplied labels* |

Reproduce (outputs are new files under the Git-ignored `outputs/`):

```bash
.venv/bin/python scripts/analyze_anpr_failures.py \
  --dataset data/processed/witvix_anpr_pilot_v1 \
  --predictions outputs/anpr/witvix_val_predictions_rapidocr_v1.jsonl \
  --plates-csv /Users/aniket/Downloads/archive/plates.csv \
  --output outputs/anpr/failure_analysis_rapidocr_v2.json \
  --details outputs/anpr/failure_details_rapidocr_v2.jsonl
```

The aggregate JSON contains no plate strings. The optional details file does and must
stay out of Git.

## 1. Failure categories (each plate gets exactly one)

| Category | Plates | Meaning |
| --- | ---: | --- |
| Correct | 37 | exact full-plate agreement |
| **Extra characters only** | **28** | the right plate plus additional characters, nothing wrong or missing |
| Mixed errors | 20 | combination of substitutions, deletions and insertions |
| Missed detection | 3 | no predicted box with IoU ≥ 0.5 |
| Look-alike substitutions only | 3 | only O/0, I/1, B/8, S/5, Z/2, G/6, A/4, D/0, Q/0, L/1 swaps |
| Reordered | 1 | correct characters in the wrong order (line order) |
| Fragment | 1 | a contiguous piece of the plate, ≥ 2 characters missing |
| Dropped characters only | 1 | deletions only |
| Empty read | 0 | box found but no text |

**The largest failure group is not misreading — it is reading too much.** 28 of the 57
non-exact plates (49%) contain every reference character in order, plus extra characters.

## 2. What the extra characters are

Most-inserted characters across all plates: `N` 20, `D` 16, `I` 15, `.` 12, `O` 4, `·` 3.

- The letters `I`, `N`, `D` appear as the token **`IND`** at the **start** of 16 reads.
  This is the country code printed on Indian high-security registration plates; it is
  not part of the registration. No validation reference contains `IND` (none contains
  the letter `I` at all).
- `.`, `·`, `:`, `°` are punctuation/marks read from bolts, borders or the emblem.
  Current normalization deliberately removes only whitespace and hyphens, so these remain.

Diagnostic counterfactual on the *same* frozen reads (**not applied** to the pipeline):

| Post-OCR rule | Exact agreement |
| --- | ---: |
| Current normalization (frozen) | 37 / 94 (39.4%) |
| + remove non-alphanumeric characters | 47 / 94 (50.0%) |
| + also remove a leading `IND` token | 58 / 94 (61.7%) |

This is the clearest available improvement, but it was found **on validation**. Before
any claim, the rule must be written as an explicit, raw-text-preserving policy (the raw
OCR string stays in the response; the removed token is recorded with a reason code),
frozen, and confirmed on independent data — the held-out test split after freeze, and
the planned car collection. A real plate segment can never be removed silently.

## 3. Conditions and plate size

| Group | Plates | Exact | CER |
| --- | ---: | ---: | ---: |
| Day / other | 65 | 30 (46.2%) | 17.9% |
| Low light | 29 | 7 (24.1%) | 38.3% |
| Single-line layout | 31 | 15 (48.4%) | 15.9% |
| Two-line layout | 63 | 22 (34.9%) | 28.1% |
| Reference box height < 40 px | 16 | 9 | 14.6% |
| 40–79 px | 33 | 18 | 15.7% |
| ≥ 80 px | 45 | 10 | 33.6% |

- **Low light** is the hardest condition: 12 of its 22 failures are mixed errors, i.e.
  genuine misreads rather than extra tokens. This needs better input, not text cleanup.
- **Large plates read worse than small ones.** 19 of the ≥ 80 px failures are
  extra-character cases: close-up crops resolve the `IND` emblem and fastener marks that
  small crops blur away. Small crops are *not* the dominant failure on this data.
- Layout rows use per-plate source labels (`plates.csv`), matching the earlier comparison
  report. Category-by-layout counts are in the JSON (`by_layout`, `layout_x_condition`).

## 4. Character confusions

- Most frequent substitution: digit `0` read as letter `O` (5). Other substitutions
  occur 1–2 times each (e.g. `1→A`, `3→N`).
- Most deleted characters are digits (`1` 9, `4` 8, `0` 7, `7` 7, `9` 6) — dropped
  digits, mostly within mixed-error reads.
- For same-length reads (44 plates), errors are spread across positions 0–9 at 1–4 each;
  no single plate position dominates.

`0→O` is where a position-aware format check helps: in `IN_COMMON_V1`, the final four
characters are digits, so an `O` there can be **flagged as a likely `0`** for the
reviewer. It must stay a suggestion; normalization never substitutes characters.

## 5. Answer for the jury: "Where does OCR fail?"

1. **Extra non-registration text** (the `IND` emblem and punctuation) — about half of
   all failures. Fixable with an explicit, auditable post-OCR rule; validation suggests
   a large gain that still has to be confirmed on independent data.
2. **Low-light misreads** — genuine character errors (38% CER). Needs capture
   improvements or recognizer adaptation, not text cleanup.
3. **Missed plates** — 3 of 94, a detector limit.
4. **Look-alike characters** — small (3 plates), best handled as reviewer suggestions.

## 6. Decision for the next step

Ordered by measured impact on these same 94 validation plates:

1. Implement the emblem/punctuation rule as an explicit policy with reason codes and
   tests, report validation results, then hold it frozen for the test evaluation.
2. Position-aware look-alike *suggestions* (no substitution) as part of format check v2.
3. Low-light and two-line recognition work, measured against the post-rule baseline.

Earlier plans prioritized two-line row splitting. This analysis shows only 1 reordered
and 1 fragment read, so a dedicated row splitter would address at most a few plates
and is deprioritized.
