"""Run the local ANPR image pipeline across a prepared dataset split and write frozen predictions.

Uses ``AnprImageService`` and ``build_local_runtime`` to run genuine
detector -> crop preprocessing -> EasyOCR -> IN_COMMON_V1 format validation
on every image in the selected split (default ``val``). Never silently skips
images, never substitutes ground-truth labels for OCR output, and records
a companion provenance JSON alongside the prediction JSONL file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anpr.contracts import ResultStatus
from src.anpr.dataset import (
    DatasetValidationError,
    load_manifest,
    sha256_file,
    validate_prepared_dataset,
)
from src.anpr.local_runtime import build_local_runtime
from src.anpr.pipeline import AnprImageService
from src.anpr.runtime_contracts import AnprRuntime


class BatchPredictionError(RuntimeError):
    """Raised when batch ANPR prediction generation or output validation fails."""


def _safe_output_jsonl_path(output_path: Path | str, *, repo_root: Path = REPO_ROOT) -> Path:
    root = repo_root.resolve()
    outputs_root = root / "outputs"
    raw = Path(output_path)
    if ".." in raw.parts:
        raise BatchPredictionError("Output path must not contain parent traversal ('..').")
    candidate = (
        Path(os.path.abspath(os.fspath(raw)))
        if raw.is_absolute()
        else Path(os.path.abspath(os.fspath(root / raw)))
    )
    if (
        not candidate.is_relative_to(outputs_root)
        or candidate == outputs_root
        or candidate.suffix.lower() != ".jsonl"
    ):
        raise BatchPredictionError("Output must be a new .jsonl file inside repository outputs/.")

    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise BatchPredictionError(f"Output path contains a symlink component: {current}")

    if candidate.exists() or candidate.is_symlink():
        raise BatchPredictionError(f"Prediction output file already exists: {candidate}")
    provenance_path = candidate.with_suffix(".provenance.json")
    if provenance_path.exists() or provenance_path.is_symlink():
        raise BatchPredictionError(f"Provenance output file already exists: {provenance_path}")
    return candidate


def _installed_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not_installed"


def _resolve_git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def generate_split_predictions(
    *,
    dataset_dir: Path,
    output_jsonl: Path,
    split: str = "val",
    allow_test_split: bool = False,
    minimum_confidence: float = 0.75,
    runtime_factory: Optional[Callable[[], AnprRuntime]] = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Run ANPR inference on every image in ``split`` and save predictions + provenance."""
    if split not in ("val", "test"):
        raise BatchPredictionError(f"Unsupported evaluation split: {split!r}")
    if split == "test" and not allow_test_split:
        raise BatchPredictionError(
            "Refusing to run inference on the held-out 'test' split without --allow-test-split."
        )

    target_jsonl = _safe_output_jsonl_path(output_jsonl, repo_root=repo_root)
    provenance_path = target_jsonl.with_suffix(".provenance.json")

    dataset_path = dataset_dir if dataset_dir.is_absolute() else (repo_root / dataset_dir)
    dataset_path = dataset_path.resolve()
    metadata = validate_prepared_dataset(dataset_path)
    records = [
        record
        for record in load_manifest(dataset_path / "manifest.jsonl", prepared=True)
        if record.split == split
    ]
    if not records:
        raise BatchPredictionError(f"Prepared dataset has no records for split={split!r}.")

    active_runtime_Holder: List[AnprRuntime] = []

    def _capturing_factory() -> AnprRuntime:
        rt = (runtime_factory or (lambda: build_local_runtime(repo_root=repo_root)))()
        active_runtime_Holder.append(rt)
        return rt

    service = AnprImageService(
        runtime_factory=_capturing_factory,
        minimum_confidence=minimum_confidence,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    prediction_lines: List[str] = []
    status_counts: Dict[str, int] = {status.value: 0 for status in ResultStatus}
    total_detected_plates = 0

    for record in records:
        image_file = dataset_path / record.image_path
        suffix = image_file.suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        payload = image_file.read_bytes()
        result = service.recognize_bytes(
            payload,
            media_type,
            request_id=f"{split}-{record.image_id}",
        )
        if result.status == ResultStatus.UNAVAILABLE:
            warnings_str = "; ".join(result.warnings) or "Local ANPR runtime unavailable."
            raise BatchPredictionError(
                f"Runtime returned UNAVAILABLE on {record.image_id}: {warnings_str}"
            )
        status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
        total_detected_plates += len(result.plates)

        row_obj = {
            "image_id": record.image_id,
            "result": result.model_dump(mode="json"),
        }
        prediction_lines.append(
            json.dumps(row_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        )

    completed_at = datetime.now(timezone.utc).isoformat()
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with target_jsonl.open("x", encoding="utf-8") as handle:
        handle.writelines(prediction_lines)

    predictions_sha256 = sha256_file(target_jsonl)

    runtime_details: Dict[str, Any] = {}
    if active_runtime_Holder:
        rt = active_runtime_Holder[0]
        det = rt.detector
        rec = rt.recognizer
        det_path = getattr(det, "checkpoint_path", None)
        rec_weights = getattr(rec, "weight_files", ())
        runtime_details = {
            "detector_checkpoint_path": (
                str(Path(det_path).relative_to(repo_root.resolve()))
                if isinstance(det_path, Path) and det_path.is_relative_to(repo_root.resolve())
                else (str(det_path) if det_path else None)
            ),
            "detector_checkpoint_sha256": getattr(det, "checkpoint_sha256", None),
            "detector_device": getattr(det, "device", None),
            "detector_confidence_threshold": getattr(det, "confidence_threshold", None),
            "detector_iou_threshold": getattr(det, "iou_threshold", None),
            "detector_image_size": getattr(det, "image_size", None),
            "ocr_model_dir": (
                str(Path(getattr(rec, "model_dir")).relative_to(repo_root.resolve()))
                if isinstance(getattr(rec, "model_dir", None), Path)
                and Path(getattr(rec, "model_dir")).is_relative_to(repo_root.resolve())
                else str(getattr(rec, "model_dir", ""))
            ),
            "ocr_device": getattr(rec, "device", None),
            "ocr_crop_view": getattr(rec, "crop_view", None),
            "ocr_adapted_checkpoint_sha256": getattr(rec, "recognizer_checkpoint_sha256", None),
            "ocr_languages": list(getattr(rec, "languages", ())),
            "ocr_weight_files": {
                Path(w).name: sha256_file(Path(w))
                for w in rec_weights
                if isinstance(w, (str, Path)) and Path(w).is_file()
            },
        }

    provenance: Dict[str, Any] = {
        "schema_version": "1.0",
        "split": split,
        "dataset_dir": str(dataset_path),
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "evidence_assurance": metadata.get("evidence_assurance"),
        "images_processed": len(records),
        "total_detected_plates": total_detected_plates,
        "result_status_counts": status_counts,
        "minimum_ocr_confidence_threshold": minimum_confidence,
        "predictions_path": str(target_jsonl),
        "predictions_sha256": predictions_sha256,
        "provenance_path": str(provenance_path),
        "git_sha": _resolve_git_sha(repo_root.resolve()),
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime": runtime_details,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _installed_version("torch"),
            "torchvision": _installed_version("torchvision"),
            "ultralytics": _installed_version("ultralytics"),
            "easyocr": _installed_version("easyocr"),
            "opencv_python_headless": _installed_version("opencv-python-headless"),
        },
    }
    with provenance_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Prepared ANPR dataset directory (e.g. data/processed/witvix_anpr_pilot_v1).",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to run inference on (default: val).",
    )
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="Explicit opt-in required before running inference on the held-out test split.",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.75,
        help="Minimum OCR engine score threshold for READ vs NEEDS_REVIEW (default: 0.75).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New .jsonl file path under outputs/ for frozen predictions.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = generate_split_predictions(
            dataset_dir=args.dataset,
            output_jsonl=args.output,
            split=args.split,
            allow_test_split=bool(args.allow_test_split),
            minimum_confidence=args.minimum_confidence,
        )
    except (DatasetValidationError, BatchPredictionError, ValueError, OSError) as exc:
        print(f"ANPR batch prediction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
