"""Prepare and verify an ANPR dataset with grouped, reproducible splits."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.anpr.contracts import PlateBox
from src.anpr.pipeline import MAX_IMAGE_BYTES, decode_image
from src.anpr.validation import normalize_plate_text


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
GROUP_POLICY = "capture+source+normalized_plate+raw_and_prepared_sha256_v1"
SafeId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Condition = Literal["clear", "blur", "angled", "low_light", "occluded", "other"]


class DatasetValidationError(ValueError):
    """A dataset or manifest cannot satisfy the training/evaluation contract."""


class PlateAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox: PlateBox
    text: str = Field(min_length=1, max_length=256)

    @field_validator("text")
    @classmethod
    def require_transcription(cls, value: str) -> str:
        if not normalize_plate_text(value):
            raise ValueError("Each labelled plate requires a verified text transcription.")
        return value


class ImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: SafeId
    image_path: str = Field(min_length=1)
    group_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=128)
    plates: list[PlateAnnotation] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=lambda: ["other"])

    @field_validator("group_id", "source_id")
    @classmethod
    def require_group_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Capture/source group IDs must be supplied, not inferred per image.")
        return value.strip()

    @field_validator("conditions")
    @classmethod
    def canonical_conditions(cls, value: list[Condition]) -> list[Condition]:
        return sorted(set(value)) or ["other"]


class PreparedRecord(ImageRecord):
    split: Literal["train", "val", "test"]
    source_image_path: str
    image_sha256: Sha256
    prepared_image_sha256: Sha256
    width: int = Field(strict=True, gt=0)
    height: int = Field(strict=True, gt=0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _confined_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative:
        raise DatasetValidationError("Dataset file paths must be relative without traversal.")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise DatasetValidationError("Symlinked dataset files/directories are not supported.")
    if not current.is_file():
        raise DatasetValidationError(f"Required dataset file is missing: {relative}")
    return current


def load_manifest(path: Path, *, prepared: bool = False) -> list[ImageRecord]:
    model = PreparedRecord if prepared else ImageRecord
    records = []
    seen = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = model.model_validate_json(line)
                if record.image_id.casefold() in seen:
                    raise DatasetValidationError(f"Duplicate image_id on manifest line {line_number}.")
                seen.add(record.image_id.casefold())
                records.append(record)
    except (OSError, UnicodeError, ValidationError) as exc:
        raise DatasetValidationError(f"Cannot read a valid ANPR manifest: {exc}") from exc
    if not records:
        raise DatasetValidationError("The ANPR manifest must contain labelled images.")
    return sorted(records, key=lambda record: record.image_id)


def _normalized_png(path: Path) -> tuple[bytes, int, int]:
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    if path.suffix.lower() not in media_types:
        raise DatasetValidationError("ANPR images must be JPEG/PNG.")
    with path.open("rb") as handle:
        payload = handle.read(MAX_IMAGE_BYTES + 1)
    try:
        bgr = decode_image(payload, media_types[path.suffix.lower()])
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc
    height, width = bgr.shape[:2]
    output = io.BytesIO()
    Image.fromarray(bgr[:, :, ::-1]).save(output, "PNG")
    encoded = output.getvalue()
    if len(encoded) > MAX_IMAGE_BYTES:
        raise DatasetValidationError("Normalized PNG exceeds 10 MiB; provide a smaller source image.")
    return encoded, width, height


def _annotation_key(record: dict) -> bytes:
    return _json_bytes(sorted(
        ((p["bbox"]["x1"], p["bbox"]["y1"], p["bbox"]["x2"], p["bbox"]["y2"],
          normalize_plate_text(p["text"])) for p in record["plates"])
    ))


def _components(records: list[dict]) -> list[list[dict]]:
    parents = list(range(len(records)))

    def find(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    anchors = {}
    duplicate_annotations = {}
    for index, record in enumerate(records):
        signature = _annotation_key(record)
        for kind in ("image_sha256", "prepared_image_sha256"):
            prior = duplicate_annotations.setdefault((kind, record[kind]), signature)
            if prior != signature:
                raise DatasetValidationError("Identical source images have conflicting plate annotations.")
        keys = [
            ("capture", record["group_id"]), ("source", record["source_id"]),
            ("bytes", record["image_sha256"]),
            ("pixels", record["prepared_image_sha256"]),
        ] + [("plate", normalize_plate_text(p["text"])) for p in record["plates"]]
        for key in keys:
            if key in anchors:
                parents[find(index)] = find(anchors[key])
            else:
                anchors[key] = index
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    return list(groups.values())


def _assign_splits(records: list[dict], seed: int) -> list[dict]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetValidationError("seed must be a nonnegative integer.")
    groups = _components(records)
    positive_groups = [group for group in groups if any(row["plates"] for row in group)]
    if len(positive_groups) < 3:
        raise DatasetValidationError(
            "At least three independent groups containing plates are required for train/val/test."
        )
    totals = {split: 0 for split in SPLITS}
    targets = {split: len(records) * fraction for split, fraction in SPLIT_FRACTIONS.items()}

    def order(group):
        tie = hashlib.sha256(_json_bytes([seed, sorted(row["image_id"] for row in group)])).hexdigest()
        return (-len(group), tie)

    ordered_positive = sorted(positive_groups, key=order)
    other = sorted([group for group in groups if not any(row["plates"] for row in group)], key=order)
    assignments = []
    for index, group in enumerate(ordered_positive + other):
        # Seed each split with a positive group, then minimize image-count error.
        if index < 3:
            split = SPLITS[index]
        else:
            split = min(SPLITS, key=lambda candidate: sum(
                (totals[name] + (len(group) if name == candidate else 0) - targets[name]) ** 2
                for name in SPLITS
            ))
        totals[split] += len(group)
        for row in group:
            assignments.append({
                **row, "split": split,
                "image_path": f"images/{split}/{row['image_id']}.png",
            })
    return sorted(assignments, key=lambda row: row["image_id"])


def _fingerprint(rows: list[dict], seed: int) -> str:
    return hashlib.sha256(_json_bytes({
        "schema_version": "1.0", "group_policy": GROUP_POLICY,
        "seed": seed, "records": rows,
    })).hexdigest()


def _split_counts(rows: list[dict]) -> dict:
    return {split: {
        "images": sum(row["split"] == split for row in rows),
        "plates": sum(len(row["plates"]) for row in rows if row["split"] == split),
        "negative_images": sum(not row["plates"] for row in rows if row["split"] == split),
    } for split in SPLITS}


def _yolo_labels(row: dict) -> str:
    lines = []
    for plate in row["plates"]:
        box = plate["bbox"]
        values = (
            (box["x1"] + box["x2"]) / (2 * row["width"]),
            (box["y1"] + box["y2"]) / (2 * row["height"]),
            (box["x2"] - box["x1"]) / row["width"],
            (box["y2"] - box["y1"]) / row["height"],
        )
        lines.append("0 " + " ".join(f"{value:.10f}" for value in values))
    return "".join(line + "\n" for line in lines)


def _safe_output(output_dir: Path) -> Path:
    repository = REPO_ROOT.resolve()
    processed = repository / "data" / "processed"
    output = Path(output_dir)
    if not output.is_absolute():
        output = repository / output
    if ".." in output.parts:
        raise DatasetValidationError("Output paths must not contain parent traversal.")
    try:
        relative = output.relative_to(processed)
    except ValueError as exc:
        raise DatasetValidationError("Output must be a child directory of data/processed/.") from exc
    if not relative.parts:
        raise DatasetValidationError("Output must not be data/processed/ itself.")
    current = repository
    for part in output.relative_to(repository).parts:
        current /= part
        if current.is_symlink():
            raise DatasetValidationError("Output directory components cannot be symlinks.")
    if output.exists():
        raise DatasetValidationError("Output already exists; choose a new dataset directory.")
    return output


def prepare_dataset(
    manifest_path: Path, source_root: Path, output_dir: Path, *, seed: int = 42, dry_run: bool = True,
) -> dict:
    """Audit sources read-only; optionally materialize a new, isolated YOLO dataset."""
    output = _safe_output(output_dir)
    source_root = Path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise DatasetValidationError("Source root must be an existing, non-symlink directory.")
    source_root = source_root.resolve()
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise DatasetValidationError("Source and output directories must be disjoint.")
    manifest_path = Path(manifest_path)
    records = load_manifest(manifest_path)
    audited = []
    for record in records:
        path = _confined_file(source_root, record.image_path)
        png, width, height = _normalized_png(path)
        for plate in record.plates:
            if plate.bbox.x2 > width or plate.bbox.y2 > height:
                raise DatasetValidationError(f"Plate bounds exceed image dimensions: {record.image_id}")
        audited.append({
            **record.model_dump(mode="json"), "source_image_path": record.image_path,
            "image_sha256": sha256_file(path),
            "prepared_image_sha256": hashlib.sha256(png).hexdigest(),
            "width": width, "height": height,
        })
    rows = _assign_splits(audited, seed)
    metadata = {
        "schema_version": "1.0", "group_policy": GROUP_POLICY, "seed": seed,
        "dataset_fingerprint": _fingerprint(rows, seed),
        "source_manifest_sha256": sha256_file(manifest_path),
        "split_counts": _split_counts(rows),
        "independent_groups": len(_components(rows)),
    }
    if dry_run:
        return metadata

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".anpr_stage_", dir=output.parent))
    try:
        for split in SPLITS:
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        for row in rows:
            png, _, _ = _normalized_png(_confined_file(source_root, row["source_image_path"]))
            if hashlib.sha256(png).hexdigest() != row["prepared_image_sha256"]:
                raise DatasetValidationError("A source image changed during preparation.")
            (staging / row["image_path"]).write_bytes(png)
            (staging / "labels" / row["split"] / f"{row['image_id']}.txt").write_text(
                _yolo_labels(row), encoding="utf-8"
            )
        (staging / "manifest.jsonl").write_text(
            "".join(_json_bytes(row).decode() + "\n" for row in rows), encoding="utf-8"
        )
        (staging / "anpr.yaml").write_text(yaml.safe_dump({
            "path": str(output), "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "license_plate"},
        }), encoding="utf-8")
        (staging / "preparation_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        validate_prepared_dataset(staging, expected_path=output)
        if output.exists():
            raise DatasetValidationError("Output was created during preparation; refusing to replace it.")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return metadata


def validate_prepared_dataset(dataset_dir: Path, *, expected_path: Path | None = None) -> dict:
    """Return verified metadata or raise DatasetValidationError, without loading ML models."""
    directory = Path(dataset_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise DatasetValidationError("Prepared ANPR dataset directory is missing or symlinked.")
    directory = directory.resolve()
    try:
        config = yaml.safe_load(_confined_file(directory, "anpr.yaml").read_text(encoding="utf-8"))
        expected = {
            "path": str((expected_path or directory).resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "license_plate"},
        }
        if (
            config != expected or type(config.get("nc")) is not int
            or not isinstance(config.get("names"), dict)
            or any(type(key) is not int for key in config["names"])
        ):
            raise DatasetValidationError("anpr.yaml must declare the prepared single-class license_plate dataset.")
        records = load_manifest(_confined_file(directory, "manifest.jsonl"), prepared=True)
        rows = [record.model_dump(mode="json") for record in records]
        metadata = json.loads(_confined_file(directory, "preparation_metadata.json").read_text())
        reassigned = _assign_splits(rows, metadata["seed"])
        if reassigned != rows:
            raise DatasetValidationError("Prepared split membership violates the deterministic group policy.")
        if (
            metadata.get("schema_version") != "1.0"
            or metadata.get("group_policy") != GROUP_POLICY
            or metadata.get("dataset_fingerprint") != _fingerprint(rows, metadata["seed"])
            or _json_bytes(metadata.get("split_counts")) != _json_bytes(_split_counts(rows))
            or metadata.get("independent_groups") != len(_components(rows))
        ):
            raise DatasetValidationError("Dataset metadata/fingerprint does not match the prepared manifest.")
        expected_files = set()
        for row in rows:
            image_path = _confined_file(directory, row["image_path"])
            if sha256_file(image_path) != row["prepared_image_sha256"]:
                raise DatasetValidationError("A prepared image hash does not match the manifest.")
            _, width, height = _normalized_png(image_path)
            if (width, height) != (row["width"], row["height"]):
                raise DatasetValidationError("A prepared image has unexpected dimensions.")
            if any(p["bbox"]["x2"] > width or p["bbox"]["y2"] > height for p in row["plates"]):
                raise DatasetValidationError("A prepared plate annotation exceeds the image bounds.")
            label_path = f"labels/{row['split']}/{row['image_id']}.txt"
            if _confined_file(directory, label_path).read_text() != _yolo_labels(row):
                raise DatasetValidationError("YOLO labels do not match the transcribed manifest.")
            expected_files.update((row["image_path"], label_path))
        actual_files = {
            str(path.relative_to(directory)) for folder in ("images", "labels")
            for path in (directory / folder).rglob("*") if path.is_file()
        }
        if actual_files != expected_files:
            raise DatasetValidationError("Unmanifested or missing image/label files were found.")
        return metadata
    except DatasetValidationError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise DatasetValidationError(f"Invalid prepared ANPR dataset: {exc}") from exc
