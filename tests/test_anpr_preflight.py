"""Read-only ANPR preflight: pin parsing, hash/status policy and no model loading."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("anpr_preflight", REPO / "scripts" / "anpr_preflight.py")
pf = importlib.util.module_from_spec(_spec)
sys.modules["anpr_preflight"] = pf  # dataclasses resolve their module via sys.modules
_spec.loader.exec_module(pf)


def _write(root: Path, rel: str, data: bytes) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_import_does_not_load_model_frameworks():
    import subprocess
    code = (
        "import importlib.util, sys;"
        f"s=importlib.util.spec_from_file_location('p', {str(REPO / 'scripts/anpr_preflight.py')!r});"
        "m=importlib.util.module_from_spec(s); sys.modules['p']=m; s.loader.exec_module(m);"
        "print([h for h in ('torch','ultralytics','easyocr','rapidocr_onnxruntime','cv2') if h in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_parse_pins_reads_only_exact_pins_and_canonicalizes_names():
    pins = pf.parse_pins("# c\nRapidOCR_onnxruntime==1.4.4\ntorch>=2\n-r other.txt\nnumpy == 2.5.2  # x\n")
    assert pins == {"rapidocr-onnxruntime": "1.4.4", "numpy": "2.5.2"}


def test_recorded_requirements_cover_every_required_package():
    pins = pf.parse_pins((REPO / "requirements-anpr.txt").read_text())
    required = {n for group in pf.REQUIRED_FOR.values() for n in group}
    assert required <= set(pins)


def test_asset_hash_match_mismatch_missing_and_symlink(tmp_path):
    good = _write(tmp_path, "models/a.pt", b"detector")
    _write(tmp_path, "models/b.onnx", b"tampered")
    (tmp_path / "models/link.pt").symlink_to(tmp_path / "models/a.pt")
    specs = [
        pf.AssetSpec("models/a.pt", good, True, "x"),
        pf.AssetSpec("models/b.onnx", "0" * 64, True, "x"),
        pf.AssetSpec("models/absent.pt", "0" * 64, True, "x"),
        pf.AssetSpec("models/optional.pt", "0" * 64, False, "x"),
        pf.AssetSpec("models/link.pt", good, True, "x"),
    ]
    statuses = [c.status for c in pf.check_assets(tmp_path, specs)]
    assert statuses == [pf.PASS, pf.FAIL, pf.FAIL, pf.WARN, pf.FAIL]


def test_package_policy_drift_warns_and_missing_required_fails():
    pins = {"torch": "2.13.0", "easyocr": "1.7.2", "rapidocr-onnxruntime": "1.4.4"}
    installed = {"torch": "2.14.0", "rapidocr-onnxruntime": "1.4.4"}
    by_name = {c.name: c.status for c in pf.check_packages(pins, "rapidocr", installed.get)}
    assert by_name["torch"] == pf.WARN            # drift only
    assert by_name["easyocr"] == pf.WARN          # not needed for RapidOCR
    assert by_name["rapidocr-onnxruntime"] == pf.PASS
    assert by_name["ultralytics"] == pf.FAIL      # required, unpinned here, and absent
    easy = {c.name: c.status for c in pf.check_packages(pins, "easyocr", installed.get)}
    assert easy["easyocr"] == pf.FAIL


def test_device_policy_never_silently_falls_back():
    assert pf.check_device("cpu", lambda d: pytest.fail("cpu must not probe")).status == pf.PASS
    assert pf.check_device("mps", lambda d: True).status == pf.PASS
    unavailable = pf.check_device("mps", lambda d: False)
    assert unavailable.status == pf.FAIL and "sandbox" in unavailable.detail
    assert pf.check_device("mps", lambda d: None).status == pf.FAIL


def test_dataset_check_reads_metadata_only(tmp_path):
    assert pf.check_dataset(tmp_path)[0].status == pf.WARN
    meta = {
        "dataset_fingerprint": pf.DATASET_FINGERPRINT,
        "split_counts": {k: {"images": v} for k, v in pf.DATASET_SPLIT_IMAGES.items()},
        "evidence_assurance": {"all_plate_texts_human_verified": False},
    }
    target = tmp_path / pf.DATASET_DIR
    target.mkdir(parents=True)
    (target / "preparation_metadata.json").write_text(json.dumps(meta))
    assert [c.status for c in pf.check_dataset(tmp_path)] == [pf.PASS] * 3
    meta["evidence_assurance"]["all_plate_texts_human_verified"] = True
    (target / "preparation_metadata.json").write_text(json.dumps(meta))
    assert pf.check_dataset(tmp_path)[2].status == pf.WARN


def test_busy_port_warns_instead_of_failing():
    checks = pf.check_ports({8000: "API", 3000: "dashboard"}, probe=lambda p: p == 8000)
    assert [c.status for c in checks] == [pf.WARN, pf.PASS]


def test_pins_are_single_sourced_from_launcher_and_adapter():
    pytest.importorskip("numpy")
    try:
        specs = pf.pinned_assets("rapidocr", REPO)
    except ImportError as exc:  # e.g. OpenCV absent outside the project .venv
        pytest.skip(f"ANPR adapter not importable here: {exc}")
    by_path = {s.relative_path: s for s in specs}
    assert by_path["models/anpr/anpr_witvix_detector_mps_v1_best.pt"].sha256.startswith("37eae626c753")
    assert by_path["models/rapidocr/ch_PP-OCRv4_rec_infer.onnx"].required
    assert not by_path["models/anpr_ocr/english_g2_witvix_v1.pth"].required
    assert len(specs) == 6


def test_cli_help_runs():
    import subprocess
    result = subprocess.run([sys.executable, str(REPO / "scripts/anpr_preflight.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "no model loading" in result.stdout
