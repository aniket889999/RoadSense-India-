"""Car evaluation set: double-label validation, refusal to run unfrozen, string-free summaries."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("car_eval", REPO / "scripts" / "car_eval.py")
ce = importlib.util.module_from_spec(_spec)
sys.modules["car_eval"] = ce
_spec.loader.exec_module(ce)


def row(**overrides):
    base = {"image_file": "a.jpg", "permission": "own_vehicle", "labeler_a": "AP", "label_a": "MH12AB1234",
            "labeler_b": "RK", "label_b": "mh 12 ab 1234", "adjudicated": "", "blur": "none", "angle": "frontal",
            "lighting": "day", "plate_layout": "single_line", "vehicle_type": "car", "notes": ""}
    return {**base, **overrides}


def test_valid_double_labelled_row(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    checked = ce.validate_rows([row()], tmp_path)
    assert checked["errors"] == [] and checked["stats"]["labeler_agreement"] == 1.0


def test_validation_errors_never_echo_plate_strings(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    bad = [row(label_b="MH12AB1284"),                  # disagreement without adjudication
           row(image_file="a.jpg", labeler_b="AP"),     # duplicate file, same person twice
           row(image_file="../x.jpg", permission="found_online", blur="very")]
    errors = ce.validate_rows(bad, tmp_path)["errors"]
    joined = " ".join(errors)
    assert "adjudicated text is required" in joined and "different people" in joined
    assert "duplicate" in joined and "permission" in joined and "blur" in joined and "plain" in joined
    assert "MH12AB" not in joined


def test_reference_prefers_adjudication_and_summary_groups_by_tag():
    assert ce.reference(row(label_b="MH12AB1284", adjudicated="MH12AB1234")) == "MH12AB1234"
    results = [
        {"blur": "none", "angle": "frontal", "lighting": "day", "plate_layout": "single_line", "vehicle_type": "car",
         "hit": True, "no_plate": False, "accepted": True, "accepted_correct": True},
        {"blur": "strong", "angle": "oblique_ge30", "lighting": "night", "plate_layout": "single_line",
         "vehicle_type": "car", "hit": False, "no_plate": True, "accepted": False, "accepted_correct": True},
    ]
    summary = ce.summarize(results)
    assert summary["overall"]["exact"] == 1 and summary["by_tag"]["blur"]["strong"]["no_plate"] == 1


def test_evaluate_refuses_without_frozen_confirmation(tmp_path, capsys):
    (tmp_path / "a.jpg").write_bytes(b"x")
    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ce.COLUMNS)
        writer.writeheader()
        writer.writerow(row())
    code = ce.main(["evaluate", "--manifest", str(manifest), "--images", str(tmp_path),
                    "--output", "outputs/never.json"])
    assert code == 1 and "frozen" in capsys.readouterr().err


def test_template_has_required_columns(tmp_path):
    out = tmp_path / "manifest.csv"
    assert ce.main(["template", "--output", str(out)]) == 0
    assert out.read_text().strip().split(",") == list(ce.COLUMNS)
