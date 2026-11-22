"""Time the existing review-only RapidOCR image service on validation images only.

No accuracy evaluation, label reads, image exports or reading persistence.
Model initialization, input-file I/O and HTTP/browser work are outside the warm
service latency. Stage wrappers preserve the existing service implementation.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WARMUP_RUNS = 3
DEFAULT_DATASET = "data/processed/witvix_anpr_pilot_v1"
STAGES = ("decode_ms", "detect_ms", "ocr_ms", "total_ms", "other_ms")


class BenchmarkError(ValueError):
    """The requested benchmark cannot preserve its scope or timing contract."""


def confined_path(raw: str | Path, base: Path) -> Path:
    """Reject lexical escapes/symlinks before resolving an allowed subtree."""
    base = base.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    if ".." in path.parts:
        raise BenchmarkError("Parent traversal is not allowed.")
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise BenchmarkError("Path is outside its allowed repository subtree.") from exc
    current = base
    if current.is_symlink():
        raise BenchmarkError("Symlinked paths are not allowed.")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise BenchmarkError("Symlinked paths are not allowed.")
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise BenchmarkError("Resolved path escapes its allowed subtree.")
    return resolved


def validation_images(dataset: str | Path, requested: list[str]) -> tuple[list[Path], dict]:
    directory = confined_path(dataset, ROOT / "data" / "processed")
    metadata_path = confined_path(directory / "preparation_metadata.json", directory)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fingerprint = metadata.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
        c not in "0123456789abcdef" for c in fingerprint
    ):
        raise BenchmarkError("Dataset fingerprint is missing or malformed.")
    count = metadata.get("split_counts", {}).get("val", {}).get("images")
    if type(count) is not int or count <= 0:
        raise BenchmarkError("Positive validation image count is required.")
    val_root = confined_path(directory / "images" / "val", directory)
    # Never enumerate images/test or read the all-split transcription manifest.
    available = sorted(val_root.iterdir())
    if len(available) != count:
        raise BenchmarkError("Validation directory count differs from prepared metadata.")
    for path in available:
        checked = confined_path(path, val_root)
        if checked.suffix.lower() != ".png" or not checked.is_file():
            raise BenchmarkError("Validation inputs must be regular prepared PNG files.")
    selected = []
    for raw in requested:
        if "test" in Path(raw).parts:
            raise BenchmarkError("Held-out test paths are forbidden.")
        checked = confined_path(raw, val_root)
        if checked.parent != val_root or checked not in available:
            raise BenchmarkError("Only registered validation-directory inputs are allowed.")
        selected.append(checked)
    if len(set(selected)) != len(selected):
        raise BenchmarkError("Repeated image inputs would bias the timing sample.")
    return selected or available, metadata


def output_path(raw: str | Path) -> Path:
    # Check the repository outputs chain too, not just a resolved child alias.
    path = confined_path(raw, ROOT.resolve())
    if not path.is_relative_to((ROOT / "outputs").resolve()) or path.suffix != ".json":
        raise BenchmarkError("Timing output must be a new JSON file under outputs/.")
    if path.exists():
        raise BenchmarkError("Existing output cannot be overwritten.")
    return path


def summarize(values: list[float]) -> dict:
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise BenchmarkError("Timings must be finite nonnegative numbers.")
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"median_ms": percentile(0.5), "p95_ms": percentile(0.95), "max_ms": ordered[-1]}


def measure_images(paths, runtime, synchronize, *, clock_ns=time.perf_counter_ns):
    """Wrap adapters/decode, while invoking the unchanged real image service."""
    from src.anpr import pipeline
    from src.anpr.runtime_contracts import AnprRuntime

    sample = {}
    decode = pipeline.decode_image

    def measured_decode(*args, **kwargs):
        started = clock_ns()
        value = decode(*args, **kwargs)
        sample["decode_ms"] += (clock_ns() - started) / 1_000_000
        return value

    class Detector:
        def detect(self, image):
            synchronize()
            started = clock_ns()
            value = runtime.detector.detect(image)
            # Include queued GPU work, not just asynchronous dispatch latency.
            synchronize()
            sample["detect_ms"] += (clock_ns() - started) / 1_000_000
            return value

    class Recognizer:
        def recognize(self, crop):
            started = clock_ns()
            value = runtime.recognizer.recognize(crop)
            sample["ocr_ms"] += (clock_ns() - started) / 1_000_000
            sample["ocr_calls"] += 1
            return value

    service = pipeline.AnprImageService(
        runtime_factory=lambda: AnprRuntime(detector=Detector(), recognizer=Recognizer())
    )
    rows = []
    inventory = hashlib.sha256()
    with patch.object(pipeline, "decode_image", measured_decode):
        for index in range(WARMUP_RUNS + len(paths)):
            warmup = index < WARMUP_RUNS
            path = paths[index % len(paths)] if warmup else paths[index - WARMUP_RUNS]
            # Bounded file I/O before the service timer, using its actual byte cap.
            with path.open("rb") as stream:
                payload = stream.read(pipeline.MAX_IMAGE_BYTES + 1)
            if len(payload) > pipeline.MAX_IMAGE_BYTES:
                raise BenchmarkError("Input exceeds the existing image-byte limit.")
            sample = {"decode_ms": 0.0, "detect_ms": 0.0, "ocr_ms": 0.0, "ocr_calls": 0}
            synchronize()
            started = clock_ns()
            result = service.recognize_bytes(payload, "image/png", "latency-only")
            synchronize()
            total = (clock_ns() - started) / 1_000_000
            if result.status.value == "UNAVAILABLE":
                raise BenchmarkError("Local runtime became unavailable; no timing report written.")
            if any(plate.recognition_status.value == "READ" for plate in result.plates):
                raise BenchmarkError("Selected RapidOCR runtime must remain review-only.")
            if not warmup:
                other = total - sum(sample[key] for key in STAGES[:3])
                if other < -0.000001:
                    raise BenchmarkError("Stage timing exceeds the enclosing service timer.")
                inventory.update(hashlib.sha256(payload).digest())
                rows.append({
                    "sample_index": len(rows), **sample, "total_ms": total,
                    "other_ms": max(0.0, other),
                })
            # Do not serialize or print result, plate boxes, strings or image data.
            del result, payload
    return {
        "image_count": len(rows), "warmup_runs": WARMUP_RUNS,
        "input_inventory_sha256": inventory.hexdigest(),
        "summary": {key: summarize([row[key] for row in rows]) for key in STAGES},
        "ocr_calls": sum(row["ocr_calls"] for row in rows), "samples": rows,
    }


def run_benchmark(paths, metadata, device):
    import torch
    from scripts.serve_anpr_local import build_environment
    from src.anpr.local_runtime import build_local_runtime

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise BenchmarkError("MPS unavailable; run outside the sandbox and stop instead of falling back.")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
            raise BenchmarkError("MPS fallback must not be enabled.")
    env = build_environment("rapidocr", device)
    env["YOLO_AUTOINSTALL"] = "false"
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env, clear=True))
        initialized = time.perf_counter_ns()
        runtime = build_local_runtime()
        init_ms = (time.perf_counter_ns() - initialized) / 1_000_000
        if (runtime.detector.device != device or runtime.recognizer.device != "cpu"
                or runtime.recognizer.engine_name != "rapidocr"
                or runtime.recognizer.crop_view != "resized_bgr"
                or not runtime.recognizer.review_only):
            raise BenchmarkError("Runtime does not match the selected local launcher settings.")
        sync = torch.mps.synchronize if device == "mps" else lambda: None
        measured = measure_images(paths, runtime, sync)
        model_sha = runtime.detector.checkpoint_sha256
        ocr_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in runtime.recognizer.weight_files}
    packages = ("torch", "torchvision", "ultralytics", "numpy", "opencv-python",
                "Pillow", "rapidocr-onnxruntime", "onnxruntime")
    chip = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    memory = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True))
    return {
        "schema_version": "anpr_latency_v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "dataset_fingerprint": metadata["dataset_fingerprint"], "split": "val",
        "detector_device": device, "ocr_device": "cpu", "ocr_engine": "rapidocr",
        "ocr_crop_view": "resized_bgr", "review_only": True,
        "detector_sha256": model_sha, "ocr_model_sha256": ocr_hashes,
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_initialization_ms": init_ms,
        "hardware": {"chip": chip, "memory_bytes": memory, "machine": platform.machine(),
                     "platform": platform.platform()},
        "python": platform.python_version(), "packages": {name: version(name) for name in packages},
        "timing_boundary": "Warm image service: decode through response construction; excludes file I/O, model init, HTTP and UI. OCR sums all crops per image; other includes quality, cleanup, crop copies and wrapper/synchronization overhead.",
        "percentile_method": "linear interpolation at (n-1)*q", **measured,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--image", action="append", default=[], help="Optional validation PNG; repeat to select a subset.")
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    parser.add_argument("--output", required=True, help="New timing JSON under repository outputs/.")
    args = parser.parse_args(argv)
    try:
        destination = output_path(args.output)
        paths, metadata = validation_images(args.dataset, args.image)
        report = run_benchmark(paths, metadata, args.device)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(json.dumps({"image_count": report["image_count"], "detector_device": args.device,
                          "summary": report["summary"]}, allow_nan=False))
        return 0
    except Exception as exc:
        # Do not echo adapter exceptions, payloads, result strings or input paths.
        print(f"Latency benchmark failed ({type(exc).__name__}); no successful report. Check validation-only paths, local assets, explicit device and a new outputs JSON target.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
