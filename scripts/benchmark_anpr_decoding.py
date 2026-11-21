"""Bounded training-only OCR decoder comparison using the existing local weights."""
from __future__ import annotations

import argparse
import csv
import json
import string
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.anpr.dataset import load_manifest, sha256_file
from src.anpr.detector import UltralyticsPlateDetector
from src.anpr.evaluation import evaluate_predictions
from src.anpr.ocr import EasyOcrPlateRecognizer, order_and_merge_ocr_detections
from src.anpr.pipeline import AnprImageService, decode_image
from src.anpr.runtime_contracts import AnprRuntime
from src.anpr.text_regions import plate_text_regions

warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")


def training_subset(dataset: Path, source: Path):
    with (source / "plates.csv").open() as f:
        layouts = {}
        for row in csv.DictReader(f):
            layouts.setdefault(Path(row["image"]).stem, set()).add(row["layout"])
    buckets = {}
    for record in load_manifest(dataset / "manifest.jsonl", prepared=True):
        if record.split != "train":
            continue
        # Mixed-layout images are assigned to the two-line bucket consistently.
        layout = "two_line" if "two_line" in layouts[record.image_id[7:]] else "single_line"
        key = ("low_light" if "low_light" in record.conditions else "other", layout)
        buckets.setdefault(key, []).append(record)
    return sorted([r for key in sorted(buckets) for r in buckets[key][:10]], key=lambda r: r.image_id)


class CachedDetector:
    detections = ()

    def detect(self, _image):
        return self.detections


class CandidateRecognizer:
    def __init__(self, reader, mode):
        self.reader, self.mode = reader, mode

    def recognize(self, image):
        if self.mode.startswith("regions"):
            regions = plate_text_regions(image)
            if regions:
                spans = []
                chars = string.ascii_uppercase + string.digits
                if self.mode == "regions_case":
                    chars += string.ascii_lowercase
                for x1, y1, x2, y2 in regions:
                    for box, text, score in self.reader.recognize(
                        np.ascontiguousarray(image[y1:y2, x1:x2]),
                        allowlist=chars, decoder="beamsearch", beamWidth=5, detail=1,
                    ):
                        shifted = [[float(x) + x1, float(y) + y1] for x, y in box]
                        spans.append((shifted, text, score))
                return order_and_merge_ocr_detections(spans)
        kwargs = {"detail": 1, "paragraph": False}
        if self.mode != "baseline":
            kwargs["allowlist"] = string.ascii_uppercase + string.digits
            if self.mode == "alnum_case":
                kwargs["allowlist"] += string.ascii_lowercase
        if self.mode == "upper_beam":
            kwargs.update(decoder="beamsearch", beamWidth=5)
        spans = self.reader.readtext(image, **kwargs)
        return order_and_merge_ocr_detections(spans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("baseline", "alnum_case", "upper_greedy", "upper_beam", "regions_upper", "regions_case"),
                        default=("baseline", "alnum_case", "upper_greedy", "upper_beam"))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        parser.error("Output must be a new directory under repository outputs/.")
    records = training_subset(args.dataset, args.source)
    modes = tuple(args.modes)
    output.mkdir(parents=True)
    spec = {"split": "train", "image_ids": [r.image_id for r in records],
            "candidates": list(modes), "selection": "most exact reads, then lowest CER",
            "source_manifest_sha256": sha256_file(args.dataset / "manifest.jsonl")}
    (output / "experiment.json").write_text(json.dumps(spec, indent=2) + "\n")
    detector = UltralyticsPlateDetector()
    reader = EasyOcrPlateRecognizer()._reader
    cached = CachedDetector()
    services = {m: AnprImageService(runtime_factory=lambda m=m: AnprRuntime(cached, CandidateRecognizer(reader, m))) for m in modes}
    predictions = {m: {} for m in modes}
    for n, record in enumerate(records, 1):
        payload = (args.dataset / record.image_path).read_bytes()
        image = decode_image(payload, "image/png")
        cached.detections = detector.detect(image)
        for mode, service in services.items():
            predictions[mode][record.image_id] = service.recognize_bytes(payload, "image/png", f"development-{record.image_id}")
        print(f"Training development images: {n}/{len(records)}", flush=True)
    for mode in modes:
        report = evaluate_predictions(records, predictions[mode])
        (output / f"{mode}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        with (output / f"{mode}_predictions.jsonl").open("x") as f:
            for image_id, result in predictions[mode].items():
                f.write(json.dumps({"image_id": image_id, "result": result.model_dump(mode="json")}) + "\n")
        counts = report["overall"]
        print(json.dumps({"candidate": mode, **{k: counts[k] for k in (
            "ground_truth_plates", "correct_plate_reads", "character_error_rate", "accepted_reads", "wrong_accepted_reads")}}), flush=True)


if __name__ == "__main__":
    main()
