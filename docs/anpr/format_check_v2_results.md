# ANPR format check v2 (`IN_PLATE_V2`) and issuing-office decoding — validation results

ANPR schema 1.3 adds an advisory `plate_format` block to every non-empty reading. It is
computed from the review text (`candidate_text` when cleanup applied, otherwise
`normalized_text`). Existing fields, text and the earlier decisions are unchanged.

| Field | Meaning |
| --- | --- |
| `pattern` | `STATE_SERIES` (state code, 1–2 digit RTO number, 1–3 letters, 4 digits), `BH_SERIES` (`YY BH #### XX`), or none |
| `state_code`, `state_code_known` | first two letters of a state series and whether they are a state/UT code |
| `lookalike_suggestions` | up to 5 readings formed by the *fewest* look-alike swaps (O/0, I/1, B/8, S/5, Z/2, G/6, A/4, T/7, D/0, Q/0, L/1) that fit a supported pattern. **Hints only; never applied** |
| `registration_origin` | state/UT name and the RTO office that issues the code, from a pinned public directory |

One decision rule was added and it is strictly more conservative: a state series whose
code is not a known state/UT code receives the review reason `UNKNOWN_STATE_CODE`, so it
can never be accepted automatically.

## Reference data and sources

- **State/UT codes**: current codes plus legacy codes still on older vehicles — Telangana
  reverted from `TS` to `TG` in March 2024; `DD` is the merged UT of Dadra and Nagar
  Haveli and Daman and Diu (`DN` before 2020); `OD` replaced `OR`; `UK` replaced `UA`.
  Source: Wikipedia, *Vehicle registration plates of India* (current-code list).
- **BH series**: `YY BH #### XX`, per the Ministry of Road Transport & Highways
  notification of 26 August 2021 (PIB release 1749764).
- **RTO offices**: `data/reference/in_rto_codes.json`, built deterministically by
  `scripts/build_rto_table.py` from Wikipedia, *List of Regional Transport Office
  districts in India*, revision 1383694791 (2026-11-04). **Content © Wikipedia
  contributors, CC BY-SA 4.0**; the derived table is shared under the same licence.
  Rebuilding from the saved revision reproduces the committed file byte for byte.
  - 35 state/UT sections, 1,302 RTO codes.
  - The source revision has no Andaman and Nicobar (`AN`) section: such plates show the
    state name with "RTO not in table".
  - Codes absent from the source (for example Tamil Nadu skips numbers whose digits sum
    to 8) are reported as not in table; nothing is guessed.

The issuing office describes **where a code is issued**, not a vehicle, an owner or a
registration. The response contains no owner, address, year or vehicle fields, and a
test fails if such fields are added. It is decoded from unverified OCR text, so a misread
digit decodes to a different office.

## Validation measurement

`scripts/evaluate_anpr_format_v2.py` re-assessed the frozen validation reads of both
engines (no inference, no test-split access) with the current policy.

**Reference labels:** all 94 fit `STATE_SERIES` with a known state code; 90 of 94 RTO
codes are present in the directory.

**RapidOCR review text vs reference (94 plates):**

| Review text | Exact | Wrong | Missed by detector |
| --- | ---: | ---: | ---: |
| Supported pattern, known state code | **58** | 3 | – |
| No supported pattern | 0 | 29 | – |
| State series, unknown state code | 0 | 1 | – |
| No matched detection | – | – | 3 |

When the review text fits a supported pattern with a known state code, it agreed with
the label in 58 of 61 plates (95.1%); every non-fitting reading was wrong. This makes the
format check a strong **review-triage** signal. It is validation-derived and is not used
for automatic acceptance.

**Look-alike suggestions:** RapidOCR — 3 wrong readings received suggestions; for 2 the
single suggestion was the reference. EasyOCR — 17 wrong readings received suggestions;
for 4 the single suggestion was the reference. No correct reading received a suggestion.

**Acceptance is unchanged.** EasyOCR (adapted) accepted 10 readings with 0 wrong under
both the frozen and the current policy; RapidOCR remains review-only (0 accepted).

**Issuing office on wrong readings:** wrong readings almost never fit a supported
pattern, so no office is decoded for them (RapidOCR: 0 of 33 wrong readings decoded to
the reference office). The decoding adds context for good reads; it does not recover
information from bad ones.

## How to present this

- "After reading the plate, the system decodes the issuing state and RTO office from a
  public directory — for example `HR55…` → Haryana · HR-55 · Gurugram (commercial
  vehicles)." Say it is a public office directory, not a registry lookup.
- "Format v2 separates likely-correct from certainly-wrong readings: on validation, 58 of
  61 pattern-fitting readings were exact and 0 of 29 non-fitting ones." Add that this is
  validation evidence on mostly truck images with publisher labels.
- The rule set and directory are not complete official catalogues; special formats
  (diplomatic, temporary, defence) remain unsupported and reviewable.
