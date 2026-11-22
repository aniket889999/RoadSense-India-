"""Risk/coverage statistics used to report automatic-acceptance evidence."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("measure_anpr_acceptance", REPO / "scripts" / "measure_anpr_acceptance.py")
ma = importlib.util.module_from_spec(_spec)
sys.modules["measure_anpr_acceptance"] = ma
_spec.loader.exec_module(ma)


@pytest.mark.parametrize("errors, n, expected", [
    (0, 10, 1 - 0.05 ** (1 / 10)),   # exact zero-failure bound (~0.259)
    (0, 50, 1 - 0.05 ** (1 / 50)),   # ~0.058, near the "rule of three" 3/50
    (1, 20, 0.2161),                 # Clopper-Pearson one-sided 95%
])
def test_upper_bound_95(errors, n, expected):
    assert ma.upper_bound_95(errors, n) == pytest.approx(expected, abs=5e-4)


def test_upper_bound_edge_cases():
    assert ma.upper_bound_95(0, 0) is None
    assert ma.upper_bound_95(5, 5) == 1.0


def test_risk_coverage_counts_unmatched_and_wrong_reads_and_respects_gate():
    rows = [
        {"score": 0.99, "gate": True, "correct": True},
        {"score": 0.95, "gate": True, "correct": False},
        {"score": 0.90, "gate": False, "correct": False},
        {"score": 0.80, "gate": True, "correct": True},
        {"score": None, "gate": True, "correct": True},   # no score: never accepted
    ]
    score_only = ma.risk_coverage(rows, total_plates=10, gate=False)
    assert [(r["threshold"], r["accepted"], r["wrong_accepted"]) for r in score_only] == [
        (0.8, 4, 2), (0.9, 3, 2), (0.95, 2, 1), (0.99, 1, 0)]
    gated = ma.risk_coverage(rows, total_plates=10, gate=True)
    assert [(r["threshold"], r["accepted"], r["wrong_accepted"]) for r in gated] == [
        (0.8, 3, 1), (0.95, 2, 1), (0.99, 1, 0)]
    best = ma.summarize(gated)["max_coverage_with_zero_wrong"]
    assert (best["threshold"], best["accepted"], best["coverage"]) == (0.99, 1, 0.1)


def test_cli_has_no_test_split_option():
    result = subprocess.run([sys.executable, str(REPO / "scripts/measure_anpr_acceptance.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "--split" not in result.stdout and "--allow-test-split" not in result.stdout
