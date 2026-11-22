"""Adapt the local English CTC recognizer using single-line training plates only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

from src.anpr.dataset import load_manifest, sha256_file
from src.anpr.dataset import validate_prepared_dataset
from src.anpr.evaluation import character_edit_distance
from src.anpr.ocr import EasyOcrPlateRecognizer
from src.anpr.validation import normalize_plate_text


def collect_samples(dataset: Path, source: Path):
    with (source / "plates.csv").open() as f:
        layouts = {(Path(r["image"]).stem, int(r["plate_index"])): r["layout"] for r in csv.DictReader(f)}
    samples = []
    for record in load_manifest(dataset / "manifest.jsonl", prepared=True):
        if record.split != "train":
            continue
        for index, plate in enumerate(record.plates):
            if layouts[(record.image_id[7:], index)] == "single_line":
                samples.append((record, plate))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--execute", action="store_true", help="Explicitly start OCR adaptation; otherwise only validate data and report counts.")
    parser.add_argument("--device", choices=("mps",), default="mps", help="This adaptation tool uses MPS model operations and CPU CTC loss explicitly.")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        parser.error("Output must be a new repository outputs/ directory.")
    if args.epochs <= 0:
        parser.error("Epochs must be positive.")
    metadata = validate_prepared_dataset(args.dataset)
    samples = collect_samples(args.dataset, args.source)
    if not samples:
        parser.error("No single-line training plates are available.")
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "single_line_train_crops": len(samples), "dataset_fingerprint": metadata["dataset_fingerprint"]}))
        return
    output.mkdir(parents=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    reader = EasyOcrPlateRecognizer(device=args.device, model_dir="models/easyocr", recognizer_weights="", recognizer_sha256="")._reader
    model = reader.recognizer
    converter = reader.converter
    device = next(model.parameters()).device
    from easyocr.recognition import AlignCollate
    from easyocr.config import imgH
    collate = AlignCollate(imgH=imgH, imgW=320, keep_ratio_with_pad=True)
    image_ids = sorted({record.image_id for record, _ in samples}, key=lambda v: hashlib.sha256(("42:" + v).encode()).hexdigest())
    holdout_ids = set(image_ids[:max(1, round(len(image_ids) * 0.20))])
    groups = {"train": [], "development": []}
    for record, plate in samples:
        image = cv2.imread(str(args.dataset / record.image_path))
        b = plate.bbox
        crop = image[math.floor(b.y1):math.ceil(b.y2), math.floor(b.x1):math.ceil(b.x2)]
        gray = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        label = normalize_plate_text(plate.text)
        if any(c not in converter.dict for c in label):
            raise ValueError("A supplied training label is outside the recognizer vocabulary.")
        groups["development" if record.image_id in holdout_ids else "train"].append((gray, label))
    spec = {"split_used": "train", "seed": 42, "epochs": args.epochs,
            "training_crops": len(groups["train"]), "internal_development_crops": len(groups["development"]),
            "internal_development_image_ids": sorted(holdout_ids), "device": str(device),
            "ctc_loss_device": "cpu (explicit differentiable transfer; MPS CTC is not used)",
            "input_height": imgH, "input_width": 320, "frozen": "FeatureExtraction",
            "base_ocr_sha256": sha256_file(ROOT / "models/easyocr/english_g2.pth"),
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "training_image_ids": sorted(set(image_ids) - holdout_ids),
            "learning_rate": 1e-4, "batch_size": 8, "weight_decay": 1e-4,
            "source_manifest_sha256": sha256_file(args.dataset / "manifest.jsonl")}
    (output / "experiment.json").write_text(json.dumps(spec, indent=2) + "\n")
    network = model.module if hasattr(model, "module") else model
    for parameter in network.FeatureExtraction.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)

    def assess():
        model.eval()
        exact, edits, chars = 0, 0, 0
        with torch.no_grad():
            for start in range(0, len(groups["development"]), 8):
                batch = groups["development"][start:start + 8]
                images = collate([item[0] for item in batch]).to(device)
                logits = model(images, torch.zeros(len(batch), 26, dtype=torch.long, device=device))
                indices = logits.argmax(2).cpu()
                texts = converter.decode_greedy(indices.reshape(-1).numpy(), torch.IntTensor([indices.shape[1]] * len(batch)))
                for observed, (_, expected) in zip(texts, batch):
                    observed = normalize_plate_text(observed)
                    exact += observed == expected
                    edits += character_edit_distance(expected, observed)
                    chars += len(expected)
        return {"correct": exact, "plates": len(groups["development"]), "cer": edits / chars}

    initial = assess()
    print(json.dumps({"epoch": 0, **initial}), flush=True)
    history = [{"epoch": 0, **initial}]
    best = (initial["correct"], -initial["cer"])
    for epoch in range(1, args.epochs + 1):
        model.train()
        network.FeatureExtraction.eval()
        order = list(range(len(groups["train"])))
        random.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), 8):
            batch = [groups["train"][i] for i in order[start:start + 8]]
            images = collate([item[0] for item in batch]).to(device)
            targets, lengths = converter.encode([item[1] for item in batch])
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, torch.zeros(len(batch), 26, dtype=torch.long, device=device))
            log_probs = logits.float().log_softmax(2).transpose(0, 1).cpu()
            loss = criterion(log_probs, targets.cpu(), torch.full((len(batch),), logits.shape[1], dtype=torch.long), lengths.cpu())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CTC loss; no optimizer step executed.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        result = {"epoch": epoch, "loss": total_loss, **assess()}
        history.append(result)
        print(json.dumps(result), flush=True)
        score = (result["correct"], -result["cer"])
        if score > best:
            best = score
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / "best.pth")
            (output / "best_epoch.json").write_text(json.dumps(result, indent=2) + "\n")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
