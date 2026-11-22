"""Named automatic-acceptance policies for the local ANPR runtime.

``review_only`` (default): RapidOCR scores are withheld and every non-empty reading needs
human review.

``rapidocr_gated_v1`` (opt-in): RapidOCR keeps its score and a reading is accepted
(READ) only if ALL of these hold, using the existing decision rules in
``src.anpr.validation.assess_ocr_read``:

* RapidOCR score >= 0.90 (RapidOCR's own scale; the EasyOCR 0.75 threshold is not reused);
* the normalized text matches IN_COMMON_V1 and the state code is a known state/UT code;
* no candidate cleanup was applied (cleaned readings always stay in review).

The 0.90 threshold was chosen on the VALIDATION split with a margin above the highest
wrong reading's score (see docs/anpr/acceptance_policy_results.md). It is validation-
derived and must be confirmed once on the held-out test split before any claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.anpr.runtime_contracts import RuntimeUnavailable

ENV_ACCEPTANCE = "ROADSENSE_ANPR_ACCEPTANCE"
REVIEW_ONLY = "review_only"
RAPIDOCR_GATED_V1 = "rapidocr_gated_v1"
RAPIDOCR_GATED_V1_MIN_SCORE = 0.90


@dataclass(frozen=True)
class AcceptanceSettings:
    policy: str
    review_only: bool
    minimum_confidence: float | None


def resolve_acceptance(engine: str, value: str | None) -> AcceptanceSettings:
    """Map an engine and policy name to recognizer/runtime settings; reject unsafe combinations."""
    policy = (value or REVIEW_ONLY).strip().lower()
    if policy == REVIEW_ONLY:
        # RapidOCR stays review-only; EasyOCR keeps its historical 0.75 service threshold.
        return AcceptanceSettings(policy, review_only=(engine == "rapidocr"), minimum_confidence=None)
    if policy == RAPIDOCR_GATED_V1:
        if engine != "rapidocr":
            raise RuntimeUnavailable(f"{RAPIDOCR_GATED_V1} applies only to the rapidocr engine.")
        return AcceptanceSettings(policy, review_only=False, minimum_confidence=RAPIDOCR_GATED_V1_MIN_SCORE)
    raise RuntimeUnavailable(
        f"Unsupported {ENV_ACCEPTANCE} {value!r}. Supported values: {REVIEW_ONLY}, {RAPIDOCR_GATED_V1}."
    )
