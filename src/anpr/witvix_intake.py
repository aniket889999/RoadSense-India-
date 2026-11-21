"""Witvix Indian Truck License Plates (LPD+LPR) read-only dataset intake and validation.

Audits the raw Witvix directory without modifying source files, validates
file confinement, pairing, decoded image dimensions, finite in-bounds bounding
boxes, nonempty plate transcriptions, and CSV / COCO / YOLO agreement, and
constructs ``src.anpr.dataset.ImageRecord`` rows with explicit grouping and
label provenance metadata.

The intake report deliberately omits all plate registration strings so no
vehicle identifiers are printed in CLI logs or summary outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from src.anpr.contracts import FormatStatus, PlateBox
from src.anpr.dataset import (
    DatasetValidationError,
    ImageRecord,
    PlateAnnotation,
    REPO_ROOT,
    sha256_file,
)
from src.anpr.validation import check_plate_format, normalize_plate_text

METADATA_CSV_COLUMNS = ("image", "width", "height", "n_plates", "state_codes", "light")
PLATES_CSV_COLUMNS = (
    "image",
    "plate_index",
    "crop",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "plate_text",
    "state_code",
    "layout",
)

ALLOWED_LIGHT_VALUES = {"day", "night"}
ALLOWED_LAYOUT_VALUES = {"single_line", "two_line"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Maximum pixel discrepancy allowed between integer CSV boxes, float COCO boxes,
# and normalized YOLO label boxes (COCO/YOLO are float; CSV rounds to integer pixels).
CSV_COCO_TOLERANCE_PX = 0.6
CSV_YOLO_TOLERANCE_PX = 0.65
COCO_YOLO_TOLERANCE_PX = 0.05

_SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")


def _require_non_symlink_dir(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(os.fspath(path)))
    try:
        info = os.lstat(raw)
    except OSError as exc:
        raise DatasetValidationError(f"{label} is missing or unreadable: {raw} ({exc})") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise DatasetValidationError(f"{label} must be an existing, non-symlink directory: {raw}")
    return raw


def _confined_source_file(source_root: Path, relative: str, label: str = "source file") -> Path:
    if not isinstance(relative, str) or not relative or relative != relative.strip():
        raise DatasetValidationError(f"Invalid empty or untrimmed path for {label}.")
    if "\\" in relative:
        raise DatasetValidationError(f"Backslash path separators are not allowed in {label}: {relative!r}")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise DatasetValidationError(f"{label} must be relative to source_root without traversal: {relative!r}")

    current = source_root
    parts = rel.parts
    for idx, part in enumerate(parts):
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise DatasetValidationError(f"Required {label} is missing: {relative}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise DatasetValidationError(f"Symlinked {label} is not allowed: {relative}")
        is_last = idx == len(parts) - 1
        if not is_last and not stat.S_ISDIR(info.st_mode):
            raise DatasetValidationError(f"Parent component of {label} is not a directory: {relative}")
        if is_last and not stat.S_ISREG(info.st_mode):
            raise DatasetValidationError(f"Required {label} is not a regular file: {relative}")

    return current


def _confined_source_subdir(source_root: Path, subdir_name: str) -> Path:
    subdir = source_root / subdir_name
    try:
        info = os.lstat(subdir)
    except OSError as exc:
        raise DatasetValidationError(f"Required directory '{subdir_name}/' is missing under {source_root}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise DatasetValidationError(f"'{subdir_name}/' must be a real, non-symlink directory under {source_root}")
    return subdir


def _list_regular_files_in_subdir(source_root: Path, subdir_name: str) -> Dict[str, Path]:
    subdir = _confined_source_subdir(source_root, subdir_name)
    files: Dict[str, Path] = {}
    with os.scandir(subdir) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            if entry.name.startswith("."):
                continue
            entry_path = subdir / entry.name
            try:
                info = os.lstat(entry_path)
            except OSError as exc:
                raise DatasetValidationError(f"Unable to stat {entry_path}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise DatasetValidationError(
                    f"Symlinked file found inside {subdir_name}/: {entry.name}"
                )
            if not stat.S_ISREG(info.st_mode):
                raise DatasetValidationError(
                    f"Non-regular file entry found inside {subdir_name}/: {entry.name}"
                )
            files[entry.name] = entry_path
    return files


def _inspect_image_dimensions(image_path: Path) -> Tuple[int, int]:
    if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise DatasetValidationError(f"Unsupported image extension: {image_path.name}")
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img2:
            normalized = ImageOps.exif_transpose(img2)
            width, height = normalized.size
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise DatasetValidationError(f"Corrupt or unreadable image {image_path.name}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise DatasetValidationError(f"Non-positive image dimensions in {image_path.name}")
    return width, height


def _parse_int_field(raw: Any, field_name: str, context: str, *, allow_zero: bool = False) -> int:
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetValidationError(f"Missing integer field '{field_name}' in {context}.")
    text = raw.strip()
    if not re.fullmatch(r"-?[0-9]+", text):
        raise DatasetValidationError(f"Invalid integer '{field_name}'={raw!r} in {context}.")
    val = int(text)
    if allow_zero and val < 0:
        raise DatasetValidationError(f"Field '{field_name}' must be >= 0 in {context}.")
    if not allow_zero and val <= 0:
        raise DatasetValidationError(f"Field '{field_name}' must be > 0 in {context}.")
    return val


def _parse_yolo_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Tuple[float, float, float, float]]:
    text = label_path.read_text(encoding="utf-8")
    boxes: List[Tuple[float, float, float, float]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise DatasetValidationError(
                f"YOLO label {label_path.name}:{line_no} must have 5 tokens ('0 cx cy w h')."
            )
        if parts[0] != "0":
            raise DatasetValidationError(
                f"YOLO label {label_path.name}:{line_no} must have class 0 ('license_plate'), got {parts[0]!r}."
            )
        try:
            cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError as exc:
            raise DatasetValidationError(
                f"YOLO label {label_path.name}:{line_no} contains non-numeric coordinates."
            ) from exc
        if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
            raise DatasetValidationError(
                f"YOLO label {label_path.name}:{line_no} contains non-finite coordinates."
            )
        if bw <= 0.0 or bh <= 0.0 or cx <= 0.0 or cy <= 0.0 or cx >= 1.0 or cy >= 1.0:
            raise DatasetValidationError(
                f"YOLO label {label_path.name}:{line_no} normalized box is out of range."
            )
        x1 = (cx - 0.5 * bw) * image_width
        y1 = (cy - 0.5 * bh) * image_height
        x2 = (cx + 0.5 * bw) * image_width
        y2 = (cy + 0.5 * bh) * image_height
        boxes.append((x1, y1, x2, y2))
    return boxes


def _load_and_validate_coco(
    coco_path: Path,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[Dict[str, Any]]]]:
    try:
        payload = json.loads(coco_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Unable to parse annotations_coco.json: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise DatasetValidationError("annotations_coco.json must contain a JSON object.")

    categories = payload.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise DatasetValidationError("annotations_coco.json must declare exactly one category.")
    cat = categories[0]
    if not isinstance(cat, Mapping) or cat.get("name") != "license_plate" or cat.get("id") != 1:
        raise DatasetValidationError(
            "annotations_coco.json category must be {'id': 1, 'name': 'license_plate'}."
        )

    images_raw = payload.get("images")
    annotations_raw = payload.get("annotations")
    if not isinstance(images_raw, list) or not isinstance(annotations_raw, list):
        raise DatasetValidationError("annotations_coco.json must contain 'images' and 'annotations' lists.")

    id_to_filename: Dict[int, str] = {}
    coco_image_dims: Dict[str, Tuple[int, int]] = {}
    for img in images_raw:
        if not isinstance(img, Mapping):
            raise DatasetValidationError("Invalid image entry in annotations_coco.json.")
        img_id = img.get("id")
        file_name = img.get("file_name")
        width = img.get("width")
        height = img.get("height")
        if (
            isinstance(img_id, bool)
            or not isinstance(img_id, int)
            or not isinstance(file_name, str)
            or not file_name.startswith("images/")
            or "/" in file_name[len("images/") :]
            or isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise DatasetValidationError(f"Malformed image entry in annotations_coco.json: {img!r}")
        short_name = file_name[len("images/") :]
        if img_id in id_to_filename or short_name in coco_image_dims:
            raise DatasetValidationError("Duplicate image id or file_name in annotations_coco.json.")
        id_to_filename[img_id] = short_name
        coco_image_dims[short_name] = (width, height)

    coco_anns_by_image: Dict[str, List[Dict[str, Any]]] = {name: [] for name in coco_image_dims}
    for ann in annotations_raw:
        if not isinstance(ann, Mapping):
            raise DatasetValidationError("Invalid annotation entry in annotations_coco.json.")
        img_id = ann.get("image_id")
        if img_id not in id_to_filename:
            raise DatasetValidationError(f"COCO annotation references unknown image_id: {img_id!r}")
        if ann.get("category_id") != 1:
            raise DatasetValidationError("COCO annotation has unexpected category_id.")
        bbox = ann.get("bbox")
        attrs = ann.get("attributes")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v)) for v in bbox)
            or not isinstance(attrs, Mapping)
        ):
            raise DatasetValidationError("Malformed COCO annotation bbox or attributes.")
        bx, by, bw, bh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if bw <= 0.0 or bh <= 0.0 or bx < 0.0 or by < 0.0:
            raise DatasetValidationError("COCO annotation bbox has non-positive width/height or negative origin.")
        coco_anns_by_image[id_to_filename[img_id]].append(
            {
                "x1": bx,
                "y1": by,
                "x2": bx + bw,
                "y2": by + bh,
                "plate_text": attrs.get("plate_text"),
                "state_code": attrs.get("state_code"),
                "layout": attrs.get("layout"),
            }
        )

    return coco_image_dims, coco_anns_by_image


def build_witvix_records(source_root: Path) -> Tuple[List[ImageRecord], Dict[str, Any]]:
    """Audit a Witvix LPD+LPR directory and return ``(records, report_dict)``.

    Does not write or modify any files. All registration text strings are
    verified and populated into the returned ``ImageRecord`` objects, while the
    returned ``report_dict`` contains only aggregate statistics, hashes, and
    provenance disclosures (zero plate registration numbers).
    """
    root = _require_non_symlink_dir(source_root, "Witvix source_root")

    metadata_csv_path = _confined_source_file(root, "metadata.csv", "metadata.csv")
    plates_csv_path = _confined_source_file(root, "plates.csv", "plates.csv")
    coco_json_path = _confined_source_file(root, "annotations_coco.json", "annotations_coco.json")

    image_files = _list_regular_files_in_subdir(root, "images")
    label_files = _list_regular_files_in_subdir(root, "labels")
    crop_files = _list_regular_files_in_subdir(root, "plates")

    if not image_files:
        raise DatasetValidationError("Witvix images/ directory contains no image files.")

    # 1. Read and validate metadata.csv
    metadata_rows: Dict[str, Dict[str, Any]] = {}
    with metadata_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METADATA_CSV_COLUMNS:
            raise DatasetValidationError(
                f"metadata.csv header mismatch: expected {METADATA_CSV_COLUMNS}, got {reader.fieldnames!r}"
            )
        for line_no, row in enumerate(reader, start=2):
            img_name = (row.get("image") or "").strip()
            if not img_name or "/" in img_name or "\\" in img_name:
                raise DatasetValidationError(f"Invalid image filename in metadata.csv line {line_no}.")
            if img_name in metadata_rows:
                raise DatasetValidationError(f"Duplicate image entry in metadata.csv: {img_name}")
            width = _parse_int_field(row.get("width"), "width", f"metadata.csv:{line_no}")
            height = _parse_int_field(row.get("height"), "height", f"metadata.csv:{line_no}")
            n_plates = _parse_int_field(row.get("n_plates"), "n_plates", f"metadata.csv:{line_no}")
            light = (row.get("light") or "").strip()
            if light not in ALLOWED_LIGHT_VALUES:
                raise DatasetValidationError(
                    f"Unsupported light condition {light!r} in metadata.csv:{line_no}."
                )
            state_codes_raw = (row.get("state_codes") or "").strip()
            if not state_codes_raw:
                raise DatasetValidationError(f"Empty state_codes in metadata.csv:{line_no}.")
            metadata_rows[img_name] = {
                "width": width,
                "height": height,
                "n_plates": n_plates,
                "state_codes": [s.strip() for s in state_codes_raw.split(",") if s.strip()],
                "light": light,
            }

    # 2. Verify exact 1:1 pairing between metadata.csv, images/, and labels/
    if set(metadata_rows.keys()) != set(image_files.keys()):
        raise DatasetValidationError(
            "Mismatch between metadata.csv image list and files inside images/."
        )

    expected_labels = {f"{Path(img_name).stem}.txt" for img_name in metadata_rows}
    if len(expected_labels) != len(metadata_rows):
        raise DatasetValidationError("Image filenames in metadata.csv have colliding stems.")
    if set(label_files.keys()) != expected_labels:
        raise DatasetValidationError(
            "Mismatch between image stems and YOLO label files inside labels/."
        )

    # 3. Read and validate plates.csv
    plates_by_image: Dict[str, List[Dict[str, Any]]] = {name: [] for name in metadata_rows}
    referenced_crops: set[str] = set()
    total_plate_rows = 0

    with plates_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PLATES_CSV_COLUMNS:
            raise DatasetValidationError(
                f"plates.csv header mismatch: expected {PLATES_CSV_COLUMNS}, got {reader.fieldnames!r}"
            )
        for line_no, row in enumerate(reader, start=2):
            img_name = (row.get("image") or "").strip()
            if img_name not in metadata_rows:
                raise DatasetValidationError(
                    f"plates.csv:{line_no} references unknown image {img_name!r}."
                )
            plate_index = _parse_int_field(
                row.get("plate_index"),
                "plate_index",
                f"plates.csv:{line_no}",
                allow_zero=True,
            )
            crop_name = (row.get("crop") or "").strip()
            if not crop_name or crop_name not in crop_files:
                raise DatasetValidationError(
                    f"plates.csv:{line_no} references missing crop file {crop_name!r}."
                )
            if crop_name in referenced_crops:
                raise DatasetValidationError(
                    f"plates.csv:{line_no} references duplicate crop file {crop_name!r}."
                )
            referenced_crops.add(crop_name)

            x_min = _parse_int_field(row.get("x_min"), "x_min", f"plates.csv:{line_no}", allow_zero=True)
            y_min = _parse_int_field(row.get("y_min"), "y_min", f"plates.csv:{line_no}", allow_zero=True)
            x_max = _parse_int_field(row.get("x_max"), "x_max", f"plates.csv:{line_no}", allow_zero=True)
            y_max = _parse_int_field(row.get("y_max"), "y_max", f"plates.csv:{line_no}", allow_zero=True)

            raw_text = row.get("plate_text")
            if not isinstance(raw_text, str) or not raw_text.strip() or not normalize_plate_text(raw_text):
                raise DatasetValidationError(
                    f"plates.csv:{line_no} has an empty or invalid plate transcription."
                )
            state_code = (row.get("state_code") or "").strip()
            if not state_code:
                raise DatasetValidationError(f"plates.csv:{line_no} has an empty state_code.")
            layout = (row.get("layout") or "").strip()
            if layout not in ALLOWED_LAYOUT_VALUES:
                raise DatasetValidationError(
                    f"plates.csv:{line_no} has an unsupported layout {layout!r}."
                )

            plates_by_image[img_name].append(
                {
                    "plate_index": plate_index,
                    "crop": crop_name,
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max),
                    "plate_text": raw_text,
                    "state_code": state_code,
                    "layout": layout,
                }
            )
            total_plate_rows += 1

    if referenced_crops != set(crop_files.keys()):
        raise DatasetValidationError(
            "Mismatch between plates.csv crop references and crop files inside plates/."
        )

    # 4. Load COCO annotations and verify agreement across CSV, COCO, YOLO, and decoded images
    coco_image_dims, coco_anns_by_image = _load_and_validate_coco(coco_json_path)
    if set(coco_image_dims.keys()) != set(metadata_rows.keys()):
        raise DatasetValidationError(
            "Mismatch between metadata.csv images and annotations_coco.json images."
        )

    records: List[ImageRecord] = []
    light_counts = {"day": 0, "night": 0}
    layout_counts = {"single_line": 0, "two_line": 0}
    format_counts = {
        FormatStatus.MATCHED.value: 0,
        FormatStatus.NO_MATCH.value: 0,
        FormatStatus.EMPTY.value: 0,
    }
    multi_plate_images = 0
    max_csv_coco_diff_px = 0.0
    max_csv_yolo_diff_px = 0.0
    max_coco_yolo_diff_px = 0.0

    for img_name in sorted(metadata_rows.keys()):
        meta = metadata_rows[img_name]
        img_path = image_files[img_name]
        stem = Path(img_name).stem
        if not _SAFE_STEM_RE.fullmatch(stem):
            raise DatasetValidationError(f"Unsafe image stem in Witvix source: {stem!r}")

        decoded_w, decoded_h = _inspect_image_dimensions(img_path)
        if (decoded_w, decoded_h) != (meta["width"], meta["height"]):
            raise DatasetValidationError(
                f"Decoded image dimensions for {img_name} ({decoded_w}x{decoded_h}) "
                f"do not match metadata.csv ({meta['width']}x{meta['height']})."
            )
        if coco_image_dims[img_name] != (decoded_w, decoded_h):
            raise DatasetValidationError(
                f"COCO dimensions for {img_name} do not match decoded image dimensions."
            )

        csv_plates = sorted(plates_by_image[img_name], key=lambda item: item["plate_index"])
        expected_indices = list(range(len(csv_plates)))
        if [p["plate_index"] for p in csv_plates] != expected_indices:
            raise DatasetValidationError(
                f"Non-contiguous plate_index sequence for {img_name} in plates.csv."
            )
        if len(csv_plates) != meta["n_plates"]:
            raise DatasetValidationError(
                f"metadata.csv n_plates={meta['n_plates']} does not match {len(csv_plates)} rows in plates.csv for {img_name}."
            )

        coco_anns = coco_anns_by_image[img_name]
        if len(coco_anns) != len(csv_plates):
            raise DatasetValidationError(
                f"COCO annotation count ({len(coco_anns)}) does not match plates.csv ({len(csv_plates)}) for {img_name}."
            )

        yolo_boxes = _parse_yolo_label_file(label_files[f"{stem}.txt"], decoded_w, decoded_h)
        if len(yolo_boxes) != len(csv_plates):
            raise DatasetValidationError(
                f"YOLO label box count ({len(yolo_boxes)}) does not match plates.csv ({len(csv_plates)}) for {img_name}."
            )

        plate_annotations: List[PlateAnnotation] = []
        for idx, (csv_p, coco_p, yolo_box) in enumerate(zip(csv_plates, coco_anns, yolo_boxes)):
            expected_crop_name = f"{stem}_{idx}{Path(csv_p['crop']).suffix}"
            if csv_p["crop"] != expected_crop_name:
                raise DatasetValidationError(
                    f"Unexpected crop filename {csv_p['crop']!r} for {img_name} plate {idx}."
                )
            # Ensure crop image itself is decodable
            _inspect_image_dimensions(crop_files[csv_p["crop"]])

            x1, y1, x2, y2 = csv_p["x_min"], csv_p["y_min"], csv_p["x_max"], csv_p["y_max"]
            if x2 <= x1 or y2 <= y1 or x1 < 0.0 or y1 < 0.0 or x2 > decoded_w or y2 > decoded_h:
                raise DatasetValidationError(
                    f"Out-of-bounds or non-positive plate box in plates.csv for {img_name} plate {idx}."
                )

            if (
                coco_p["plate_text"] != csv_p["plate_text"]
                or coco_p["state_code"] != csv_p["state_code"]
                or coco_p["layout"] != csv_p["layout"]
            ):
                raise DatasetValidationError(
                    f"COCO attributes mismatch plates.csv on {img_name} plate {idx}."
                )

            csv_coco_diff = max(
                abs(x1 - coco_p["x1"]),
                abs(y1 - coco_p["y1"]),
                abs(x2 - coco_p["x2"]),
                abs(y2 - coco_p["y2"]),
            )
            if csv_coco_diff > CSV_COCO_TOLERANCE_PX:
                raise DatasetValidationError(
                    f"CSV vs COCO bbox discrepancy ({csv_coco_diff:.3f}px) exceeds tolerance on {img_name} plate {idx}."
                )

            yx1, yy1, yx2, yy2 = yolo_box
            csv_yolo_diff = max(abs(x1 - yx1), abs(y1 - yy1), abs(x2 - yx2), abs(y2 - yy2))
            if csv_yolo_diff > CSV_YOLO_TOLERANCE_PX:
                raise DatasetValidationError(
                    f"CSV vs YOLO bbox discrepancy ({csv_yolo_diff:.3f}px) exceeds tolerance on {img_name} plate {idx}."
                )

            coco_yolo_diff = max(
                abs(coco_p["x1"] - yx1),
                abs(coco_p["y1"] - yy1),
                abs(coco_p["x2"] - yx2),
                abs(coco_p["y2"] - yy2),
            )
            if coco_yolo_diff > COCO_YOLO_TOLERANCE_PX:
                raise DatasetValidationError(
                    f"COCO vs YOLO bbox discrepancy ({coco_yolo_diff:.4f}px) exceeds tolerance on {img_name} plate {idx}."
                )

            max_csv_coco_diff_px = max(max_csv_coco_diff_px, csv_coco_diff)
            max_csv_yolo_diff_px = max(max_csv_yolo_diff_px, csv_yolo_diff)
            max_coco_yolo_diff_px = max(max_coco_yolo_diff_px, coco_yolo_diff)

            normalized_text = normalize_plate_text(csv_p["plate_text"])
            fmt_status = check_plate_format(normalized_text)
            format_counts[fmt_status.value] += 1
            layout_counts[csv_p["layout"]] += 1

            plate_annotations.append(
                PlateAnnotation(
                    bbox=PlateBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    text=csv_p["plate_text"],
                    label_provenance="publisher_mixed",
                )
            )

        if len(plate_annotations) > 1:
            multi_plate_images += 1
        light_counts[meta["light"]] += 1
        conditions = ["low_light"] if meta["light"] == "night" else ["other"]

        records.append(
            ImageRecord(
                image_id=f"witvix-{stem}",
                image_path=f"images/{img_name}",
                group_id=f"witvix_vehicle_proxy_{stem}",
                grouping_basis="publisher_vehicle_proxy",
                source_id=f"witvix_original_{stem}",
                plates=plate_annotations,
                conditions=conditions,
            )
        )

    records.sort(key=lambda r: r.image_id)

    source_hashes: Dict[str, Optional[str]] = {
        "metadata.csv": sha256_file(metadata_csv_path),
        "plates.csv": sha256_file(plates_csv_path),
        "annotations_coco.json": sha256_file(coco_json_path),
        "README.md": (
            sha256_file(_confined_source_file(root, "README.md", "README.md"))
            if (root / "README.md").exists()
            else None
        ),
        "data.yaml": (
            sha256_file(_confined_source_file(root, "data.yaml", "data.yaml"))
            if (root / "data.yaml").exists()
            else None
        ),
    }

    report: Dict[str, Any] = {
        "dataset_source": "witvix_indian_truck_lpd_lpr_sample",
        "source_root": str(root),
        "ignored_source_split_file": "data.yaml (declares train=images and val=images identically; ignored)",
        "counts": {
            "images": len(records),
            "yolo_label_files": len(label_files),
            "plate_annotations": total_plate_rows,
            "crop_files": len(crop_files),
            "multi_plate_images": multi_plate_images,
            "single_plate_images": len(records) - multi_plate_images,
            "light_distribution": light_counts,
            "layout_distribution": layout_counts,
            "in_common_v1_format_counts": format_counts,
        },
        "geometry_agreement": {
            "csv_coco_yolo_consistent": True,
            "max_csv_coco_diff_px": round(max_csv_coco_diff_px, 4),
            "max_csv_yolo_diff_px": round(max_csv_yolo_diff_px, 4),
            "max_coco_yolo_diff_px": round(max_coco_yolo_diff_px, 6),
        },
        "source_file_sha256": source_hashes,
        "provenance_and_limitations": {
            "grouping_basis": "publisher_vehicle_proxy",
            "capture_session_disjointness": "not_established_vehicle_proxy",
            "label_provenance": "publisher_mixed",
            "publisher_reported_human_checked_subset": "341 of 589 plates (58%), unflagged per row",
            "condition_mapping_policy": (
                "night -> ['low_light']; day -> ['other'] (daylight does not prove clear/unblurred imagery)"
            ),
            "registration_numbers_redacted_in_report": True,
        },
    }
    return records, report


def _safe_interim_manifest_path(
    output_manifest: Path | str,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    root = _require_non_symlink_dir(repo_root or REPO_ROOT, "repository root")
    raw_out = Path(output_manifest)
    if ".." in raw_out.parts:
        raise DatasetValidationError("Output manifest path must not contain parent traversal ('..').")

    candidate = (
        Path(os.path.abspath(os.fspath(raw_out)))
        if raw_out.is_absolute()
        else Path(os.path.abspath(os.fspath(root / raw_out)))
    )
    interim_root = root / "data" / "interim"
    try:
        rel_to_interim = candidate.relative_to(interim_root)
    except ValueError as exc:
        raise DatasetValidationError(
            f"Output manifest must reside inside repository data/interim/: {output_manifest!r}"
        ) from exc

    if not rel_to_interim.parts or candidate.suffix.lower() != ".jsonl":
        raise DatasetValidationError(
            "Output manifest must be a .jsonl file path under data/interim/."
        )

    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                info = os.lstat(current)
            except OSError as exc:
                raise DatasetValidationError(f"Unable to inspect path component {current}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise DatasetValidationError(
                    f"Output manifest path contains a forbidden symlink component: {current}"
                )

    if candidate.exists() or candidate.is_symlink():
        raise DatasetValidationError(
            f"Output manifest already exists; refusing to overwrite: {candidate}"
        )
    return candidate


def write_witvix_manifest(
    records: Sequence[ImageRecord],
    output_manifest: Path | str,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Path, str]:
    """Write validated ``ImageRecord`` entries to a new JSONL manifest under ``data/interim/``."""
    target = _safe_interim_manifest_path(output_manifest, repo_root=repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Re-check parent chain has no symlinks after mkdir
    _safe_interim_manifest_path(target, repo_root=repo_root)

    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
        for record in sorted(records, key=lambda r: r.image_id)
    ]
    payload = "".join(lines).encode("utf-8")
    # Exclusive creation keeps the no-overwrite contract if another invocation
    # creates the file after the path preflight.
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise DatasetValidationError(f"Output manifest already exists: {target}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    return target, digest
