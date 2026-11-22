"""IN_PLATE_V2: state codes, BH series and look-alike suggestions that are never applied."""
import json
import pytest
from pydantic import ValidationError

from src.anpr.contracts import OcrAssessment, ReadStatus, ReasonCode
from src.anpr.validation import (
    LEGACY_STATE_CODES, STATE_UT_CODES, assess_ocr_read, assess_plate_format_v2, lookalike_suggestions,
)


def test_code_lists_include_current_and_legacy_telangana_and_merged_ut():
    assert {"TG", "DD", "LA", "OD", "UK"} <= STATE_UT_CODES
    assert {"TS", "DN", "OR", "UA"} == LEGACY_STATE_CODES
    assert not STATE_UT_CODES & LEGACY_STATE_CODES


@pytest.mark.parametrize("text, pattern, state, known", [
    ("MH12AB1234", "STATE_SERIES", "MH", True),
    ("TS09AB1234", "STATE_SERIES", "TS", True),      # legacy code still on the road
    ("XY12AB1234", "STATE_SERIES", "XY", False),
    ("22BH1234AA", "BH_SERIES", None, None),
    ("22BH1234A", None, None, None),                 # BH needs two final letters
    ("MH12AB12O4", None, None, None),
])
def test_patterns_and_state_codes(text, pattern, state, known):
    result = assess_plate_format_v2(text, "normalized")
    assert (result.pattern, result.state_code, result.state_code_known) == (pattern, state, known)


def test_suggestions_are_lookalike_swaps_into_supported_patterns_only():
    assert lookalike_suggestions("MH12AB12O4") == ["MH12AB1204"]
    assert "MH12AB1234" in lookalike_suggestions("MHI2AB1234")
    assert lookalike_suggestions("MH12AB1234") == []           # nothing to suggest for a fit
    assert len(lookalike_suggestions("O0O0O0O0O0")) <= 5
    for option in lookalike_suggestions("8H12A81234"):
        assert len(option) == len("8H12A81234")


def test_suggestions_never_change_text_fields_or_decisions():
    result = assess_ocr_read("MH12AB12O4", 0.99)
    assert result.normalized_text == "MH12AB12O4" and result.candidate_text is None
    assert result.plate_format.lookalike_suggestions == ["MH12AB1204"]
    assert result.recognition_status == ReadStatus.NEEDS_REVIEW
    assert result.reason_codes == [ReasonCode.FORMAT_NOT_SUPPORTED]


def test_unknown_state_code_blocks_acceptance_and_known_code_keeps_it():
    assert assess_ocr_read("MH12AB1234", 0.99).recognition_status == ReadStatus.READ
    unknown = assess_ocr_read("XY12AB1234", 0.99)
    assert unknown.recognition_status == ReadStatus.NEEDS_REVIEW
    assert unknown.reason_codes == [ReasonCode.UNKNOWN_STATE_CODE]


def test_format_v2_assesses_the_cleaned_candidate_when_present():
    result = assess_ocr_read("IND MH12AB1234", None, review_only=True)
    assert result.plate_format.assessed_text_source == "candidate"
    assert result.plate_format.pattern == "STATE_SERIES" and result.plate_format.state_code == "MH"


def test_contract_requires_unknown_state_reason_consistency():
    base = assess_ocr_read("XY12AB1234", 0.99).model_dump()
    with pytest.raises(ValidationError):
        OcrAssessment(**{**base, "reason_codes": []})
    known = assess_ocr_read("MH12AB1234", None, review_only=True).model_dump()
    with pytest.raises(ValidationError):
        OcrAssessment(**{**known, "reason_codes": [ReasonCode.OCR_REVIEW_ONLY, ReasonCode.UNKNOWN_STATE_CODE]})


def test_empty_text_has_no_format_assessment():
    assert assess_ocr_read("", None).plate_format is None


def test_measurement_script_buckets_and_suggestion_hits_without_strings():
    import importlib.util
    import sys
    from pathlib import Path
    from tests.test_anpr_evaluation import prediction, record
    from src.anpr.validation import normalize_plate_text

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_anpr_format_v2.py"
    spec = importlib.util.spec_from_file_location("evaluate_anpr_format_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_anpr_format_v2"] = module
    spec.loader.exec_module(module)

    records = [record("a"), record("b", text="MH12AB1204"), record("c")]
    predictions = {"a": prediction("MH12AB1234"), "b": prediction("MH12AB12O4"), "c": prediction("XY12AB1234")}
    result = module.analyze_engine(records, predictions, normalize_plate_text)
    table = result["review_text_by_format"]
    assert table["STATE_KNOWN"]["correct"] == 1
    assert table["NO_PATTERN"]["wrong"] == 1 and table["STATE_UNKNOWN"]["wrong"] == 1
    assert result["suggestions"]["wrong_with_reference_in_suggestions"] == 1
    assert "MH12AB1204" not in json.dumps(result) and "MH12AB1234" not in json.dumps(result)


def test_registration_origin_decodes_state_and_rto_from_pinned_table():
    from src.anpr.rto import load_table

    table = load_table()
    assert table is not None and table["source"]["license"] == "CC BY-SA 4.0"
    assert len(table["coverage"]) == 35 and len(table["entries"]) >= 1300
    origin = assess_ocr_read("HR55AZ0090", None, review_only=True).plate_format.registration_origin
    assert (origin.state_name, origin.rto_code, origin.rto_in_table) == ("Haryana", "HR-55", True)
    assert origin.rto_office == "Gurugram" and "commercial" in origin.rto_notes
    assert origin.source_revision == table["source"]["revision_id"]


def test_registration_origin_never_guesses_and_ignores_other_patterns():
    missing = assess_ocr_read("TN08AB1234", None, review_only=True).plate_format.registration_origin
    assert missing.state_name == "Tamil Nadu" and missing.rto_in_table is False and missing.rto_office is None
    legacy = assess_ocr_read("TS09AB1234", None, review_only=True).plate_format.registration_origin
    assert "legacy" in legacy.state_name
    assert assess_ocr_read("22BH1234AA", None, review_only=True).plate_format.registration_origin is None
    assert assess_ocr_read("MH12AB12O4", None, review_only=True).plate_format.registration_origin is None


def test_registration_origin_has_no_owner_or_vehicle_fields():
    from src.anpr.contracts import RegistrationOrigin

    banned = {"owner", "name", "year", "vehicle", "address", "phone"}
    assert not {f for f in RegistrationOrigin.model_fields for b in banned if b in f and f != "state_name"}
