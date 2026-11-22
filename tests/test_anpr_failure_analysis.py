"""Failure categorization on synthetic plates; validation-only and string-free aggregates."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_anpr_failures", REPO / "scripts" / "analyze_anpr_failures.py")
fa = importlib.util.module_from_spec(_spec)
sys.modules["analyze_anpr_failures"] = fa
_spec.loader.exec_module(fa)


def box(x1, y1, x2, y2):
    return NS(x1=x1, y1=y1, x2=x2, y2=y2)


@pytest.mark.parametrize("expected, observed, category", [
    ("MH12AB1234", "MH12AB1234", "CORRECT"),
    ("MH12AB1234", None, "MISSED_DETECTION"),
    ("MH12AB1234", "", "EMPTY_READ"),
    ("MH12AB1234", "AB1234MH12", "REORDERED"),
    ("MH12AB1234", "AB1234", "FRAGMENT"),
    ("MH12AB1234", "MH12A1234", "DROPPED_CHARS"),
    ("MH12AB1234", "MH12AB12340", "EXTRA_CHARS"),
    ("MH12AB1234", "MH12A81234", "CONFUSABLE_ONLY"),
    ("MH12AB1234", "MH12AX1234", "MIXED_ERRORS"),
    ("MH12AB1234", "MH12AB123", "DROPPED_CHARS"),  # 1 missing char is not a fragment
])
def test_categorize(expected, observed, category):
    assert fa.categorize(expected, observed) == category


def test_alignment_edit_count_equals_levenshtein():
    sys.path.insert(0, str(REPO))
    try:
        from src.anpr.evaluation import character_edit_distance
    except ImportError as exc:
        pytest.skip(f"evaluation module unavailable: {exc}")
    for e, o in [("MH12AB1234", "M12AB12345"), ("ABC", ""), ("", "XY"), ("KA01", "KAO1")]:
        assert sum(op != "=" for op, _, _ in fa.align(e, o)) == character_edit_distance(e, o)


def test_matching_is_greedy_iou_and_ignores_text():
    gt = [box(0, 0, 100, 40), box(200, 0, 300, 40)]
    preds = [box(205, 0, 300, 40), box(500, 500, 510, 510)]
    assert fa.match(gt, preds) == {1: 0}


def _record(image_id, split="val", text="MH12AB1234", conditions=("other",)):
    return NS(image_id=image_id, split=split, conditions=list(conditions),
              plates=[NS(text=text, bbox=box(0, 0, 100, 50))])


def test_analyze_aggregates_contain_no_plate_strings_and_split_is_enforced():
    records = [_record("witvix-a"), _record("witvix-b", conditions=("low_light",))]
    predictions = {
        "witvix-a": NS(plates=[NS(raw_text="MH12AB1234", bbox=box(0, 0, 100, 50))]),
        "witvix-b": NS(plates=[NS(raw_text="AB1234", bbox=box(0, 0, 100, 50))]),
    }
    report, details = fa.analyze(records, predictions, {"witvix-a": ["single_line"], "witvix-b": ["two_line"]}, str.upper)
    assert report["overall"]["correct"] == 1 and report["overall"]["categories"]["FRAGMENT"] == 1
    assert report["by_layout"]["two_line"]["categories"]["FRAGMENT"] == 1
    encoded = json.dumps(report)
    assert "MH12AB1234" not in encoded and "AB1234" not in encoded
    assert details[1]["observed"] == "AB1234"  # strings live only in the local details file
    with pytest.raises(ValueError):
        fa.analyze([_record("witvix-t", split="test")], {}, {}, str.upper)


def test_cli_has_no_split_option_and_outputs_must_be_new_files_under_outputs(tmp_path):
    import subprocess
    result = subprocess.run([sys.executable, str(REPO / "scripts/analyze_anpr_failures.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "--split" not in result.stdout and "VALIDATION" in result.stdout
    with pytest.raises(ValueError):
        fa._new_output(tmp_path / "x.json", ".json")
    with pytest.raises(ValueError):
        fa._new_output(Path("outputs/../escape.json"), ".json")


def test_layout_lookup_is_per_plate_and_box_confirmed(tmp_path):
    csv_path = tmp_path / "plates.csv"
    csv_path.write_text(
        "image,plate_index,x_min,y_min,x_max,y_max,layout\n"
        "img_1.jpg,0,0,0,100,50,single_line\n"
        "img_1.jpg,1,200,0,300,50,two_line\n"
        "img_2.jpg,0,10,10,60,40,two_line\n"
    )
    rec1 = NS(image_id="witvix-img_1", plates=[NS(bbox=box(0, 0, 100, 50)), NS(bbox=box(200.4, 0, 300, 50))])
    rec2 = NS(image_id="witvix-img_2", plates=[NS(bbox=box(30, 10, 60, 40))])  # box disagrees -> unknown
    rec3 = NS(image_id="witvix-img_9", plates=[NS(bbox=box(0, 0, 1, 1))])
    assert fa.load_layouts(csv_path, [rec1, rec2, rec3]) == {
        "witvix-img_1": ["single_line", "two_line"],
        "witvix-img_2": ["unknown"],
        "witvix-img_9": ["unknown"],
    }


def test_counterfactuals_are_reported_but_never_change_categories():
    records = [_record("witvix-a"), _record("witvix-b")]
    predictions = {
        "witvix-a": NS(plates=[NS(raw_text="IND MH12AB1234", bbox=box(0, 0, 100, 50))]),
        "witvix-b": NS(plates=[NS(raw_text="MH12.AB1234", bbox=box(0, 0, 100, 50))]),
    }
    normalize = lambda t: t.upper().replace(" ", "")
    report, _ = fa.analyze(records, predictions, {}, normalize)
    cf = report["diagnostic_counterfactuals"]
    assert report["overall"]["correct"] == 0 == cf["baseline_exact"]
    assert cf["strip_non_alnum"] == 1 and cf["strip_non_alnum_and_leading_IND"] == 2
    assert fa.strip_non_alnum_and_leading_emblem("MH12AB1234IND") == "MH12AB1234IND"  # only a leading token
