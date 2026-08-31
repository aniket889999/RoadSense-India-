"""Create a human-reviewed pothole curation pool from an annotation kit.

This module deliberately has no model, training, or inference dependencies.
It turns a *human-reviewed* subset of a RoadSense annotation kit into a small,
portable collection of JPEGs and YOLO class-0 labels.  It is intended for
future data curation only; it does not train a model, score a model, or modify
the original video, the frozen dataset, or a held-out test split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

from src.manual_annotations import parse_manual_csv


RECORDING_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INTEGER_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_FRAME_FILENAME_RE = re.compile(r"^frame_([0-9]{5,})\.jpg$")

_KIT_MANIFEST_HEADERS = [
    "frame_index",
    "timestamp_seconds",
    "frame_file",
    "width",
    "height",
]
_REVIEW_HEADERS = ["frame_index", "review_status", "note"]
_REVIEW_STATUSES = {"pothole_confirmed", "no_pothole_confirmed"}

_MAX_KIT_MEMBERS = 2_000
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_FRAME_BYTES = 50 * 1024 * 1024
_MAX_TOTAL_FRAME_BYTES = 500 * 1024 * 1024
_MAX_FRAME_WIDTH = 8_192
_MAX_FRAME_HEIGHT = 8_192
_MAX_FRAME_PIXELS = 36_000_000


@dataclass(frozen=True)
class KitFrame:
    """One validated, kit-supplied JPEG frame.

    ``jpeg_bytes`` is held only in memory while a curation batch is prepared;
    no original MP4 path or bytes are retained in the result.
    """

    frame_index: int
    timestamp_seconds: float
    frame_file: str
    width: int
    height: int
    jpeg_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class FrameReview:
    frame_index: int
    review_status: str
    note: str


def prepare_manual_curation_batch(
    annotation_kit_bytes: bytes,
    annotations_csv_bytes: bytes,
    frame_review_csv_bytes: bytes,
    recording_id: str,
    *,
    output_dir: str | Path | None = None,
    write: bool = False,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate and optionally export a manual curation batch.

    The only accepted labels are manual ``pothole`` boxes from the strict
    RoadSense manual CSV.  A second strict review CSV decides which sampled
    frames enter the pool.  A confirmed-positive frame must contain at least
    one manual box; a confirmed-negative frame must contain none.

    ``write=False`` is a dry run: it performs the same input and output-path
    preflight but creates no directories or files.  When ``write=True``, the
    batch is assembled under a sibling staging directory and atomically
    promoted to exactly ``data/interim/manual_curation/<recording_id>``.

    Returns ``(summary, errors)``.  Any validation failure returns ``({},
    [readable error, ...])`` and no output is written.
    """

    try:
        resolved_repo = _resolve_repo_root(repo_root)
        safe_recording_id = _validate_recording_id(recording_id)
        target_dir = _resolve_target_dir(
            resolved_repo, safe_recording_id, output_dir=output_dir
        )

        frames, kit_error = _read_annotation_kit(annotation_kit_bytes)
        if kit_error:
            return {}, [kit_error]
        if not frames:
            return {}, ["Annotation kit contains no sampled frames."]

        first_frame = frames[0]
        if any(
            frame.width != first_frame.width or frame.height != first_frame.height
            for frame in frames
        ):
            return {}, ["Annotation kit frames must all have the same dimensions."]

        allowed_frames = [frame.frame_index for frame in frames]
        reviews, review_errors = _parse_frame_review_csv(
            _as_bytes(frame_review_csv_bytes, "Frame-review CSV"),
            allowed_frames,
        )
        if review_errors:
            return {}, review_errors

        manual_csv = _as_bytes(annotations_csv_bytes, "Manual annotations CSV")
        annotations, annotation_errors = parse_manual_csv(
            manual_csv,
            first_frame.width,
            first_frame.height,
            allowed_frames,
        )
        # The application's strict manual parser correctly rejects a header-only
        # annotation file for normal report generation.  For a curation batch,
        # however, a canonical header-only file is the unambiguous way to say
        # "there are no manual boxes".  Permit only that precise representation
        # and let the explicit frame-review rules decide whether zero boxes are
        # valid for each selected frame.  All other parser errors remain
        # failures.
        if annotation_errors:
            if _is_strict_empty_manual_csv(manual_csv):
                annotations = []
            else:
                return {}, annotation_errors

        annotations_by_frame: Dict[int, List[Any]] = {}
        for annotation in annotations:
            annotations_by_frame.setdefault(annotation.frame_index, []).append(annotation)

        reviewed_indices = {review.frame_index for review in reviews}
        annotated_unreviewed = sorted(set(annotations_by_frame) - reviewed_indices)
        if annotated_unreviewed:
            return {}, [
                "Manual annotations exist for frame(s) without a frame review: "
                + ", ".join(str(index) for index in annotated_unreviewed)
                + ". Review every annotated frame before export."
            ]

        for review in reviews:
            boxes = annotations_by_frame.get(review.frame_index, [])
            if review.review_status == "pothole_confirmed" and not boxes:
                return {}, [
                    f"Frame {review.frame_index} is pothole_confirmed but has no manual pothole box."
                ]
            if review.review_status == "no_pothole_confirmed" and boxes:
                return {}, [
                    f"Frame {review.frame_index} is no_pothole_confirmed but has manual pothole box(es)."
                ]

        frame_map = {frame.frame_index: frame for frame in frames}
        selected: List[Dict[str, Any]] = []
        for review in reviews:
            frame = frame_map[review.frame_index]
            boxes = annotations_by_frame.get(review.frame_index, [])
            yolo_text = _manual_boxes_to_yolo(
                boxes, width=frame.width, height=frame.height, frame_index=frame.frame_index
            )
            # Annotation kits are normally written by OpenCV and therefore do
            # not carry EXIF.  Treat that as a useful default, not a privacy
            # guarantee: a hand-built or otherwise altered kit could.  Export
            # a freshly encoded RGB JPEG so source-image EXIF/XMP/comment
            # metadata is never copied into the curation pool.
            image_bytes = _exportable_jpeg(
                frame.jpeg_bytes,
                width=frame.width,
                height=frame.height,
                frame_name=frame.frame_file,
            )
            image_rel = f"images/{frame.frame_file}"
            label_rel = f"labels/{Path(frame.frame_file).with_suffix('.txt').name}"
            label_bytes = yolo_text.encode("utf-8")
            selected.append(
                {
                    "frame": frame,
                    "review": review,
                    "image_relpath": image_rel,
                    "label_relpath": label_rel,
                    "image_bytes": image_bytes,
                    "image_sha256": _sha256_bytes(image_bytes),
                    "label_bytes": label_bytes,
                    "label_sha256": _sha256_bytes(label_bytes),
                    "box_count": len(boxes),
                }
            )

        _validate_output_target(
            target_dir,
            resolved_repo,
            safe_recording_id,
            overwrite=overwrite,
        )

        manifest_rows = _manifest_rows(safe_recording_id, selected)
        manifest_bytes = _csv_bytes(
            [
                "recording_id",
                "frame_index",
                "timestamp_seconds",
                "review_status",
                "frame_file",
                "label_file",
                "source_image_sha256",
                "exported_image_sha256",
                "label_sha256",
                "width",
                "height",
                "box_count",
            ],
            manifest_rows,
        )
        metadata = _build_metadata(
            recording_id=safe_recording_id,
            annotation_kit_bytes=_as_bytes(annotation_kit_bytes, "Annotation kit"),
            annotations_csv_bytes=_as_bytes(annotations_csv_bytes, "Manual annotations CSV"),
            frame_review_csv_bytes=_as_bytes(frame_review_csv_bytes, "Frame-review CSV"),
            selected=selected,
            manifest_bytes=manifest_bytes,
            target_dir=target_dir,
            repo_root=resolved_repo,
            dry_run=not write,
        )
        metadata_bytes = json.dumps(
            metadata, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8")

        if write:
            _write_batch_atomically(
                target_dir=target_dir,
                repo_root=resolved_repo,
                selected=selected,
                manifest_bytes=manifest_bytes,
                metadata_bytes=metadata_bytes,
                overwrite=overwrite,
            )

        summary = {
            "mode": "manual_curation_pool",
            "recording_id": safe_recording_id,
            "dry_run": not write,
            "output_relative_path": _repo_relative_posix(target_dir, resolved_repo),
            "reviewed_frame_count": len(selected),
            "pothole_confirmed_frame_count": sum(
                entry["review"].review_status == "pothole_confirmed" for entry in selected
            ),
            "no_pothole_confirmed_frame_count": sum(
                entry["review"].review_status == "no_pothole_confirmed" for entry in selected
            ),
            "manual_pothole_box_count": sum(entry["box_count"] for entry in selected),
            "artifacts": [
                "images/",
                "labels/",
                "manifests/curation_manifest.csv",
                "manifests/curation_metadata.json",
            ],
            "selected_frames": [
                {
                    "frame_index": entry["frame"].frame_index,
                    "review_status": entry["review"].review_status,
                    "frame_file": entry["image_relpath"],
                    "label_file": entry["label_relpath"],
                    "source_image_sha256": entry["frame"].sha256,
                    "exported_image_sha256": entry["image_sha256"],
                    "label_sha256": entry["label_sha256"],
                    "box_count": entry["box_count"],
                }
                for entry in selected
            ],
        }
        return summary, []
    except ManualCurationError as exc:
        return {}, [str(exc)]
    except (OSError, ValueError, zipfile.BadZipFile, UnidentifiedImageError) as exc:
        return {}, [f"Manual curation preflight failed: {exc}"]


class ManualCurationError(ValueError):
    """Raised internally for a safe, readable curation preflight failure."""


def _as_bytes(value: bytes, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ManualCurationError(f"{label} must be bytes.")
    output = bytes(value)
    if not output:
        raise ManualCurationError(f"{label} is empty.")
    return output


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    raw = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    raw = Path(os.path.abspath(raw))
    try:
        info = os.lstat(raw)
    except OSError as exc:
        raise ManualCurationError(f"Repository root is unavailable: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ManualCurationError("Repository root must be a real local directory, not a symlink.")
    return raw


def _validate_recording_id(recording_id: str) -> str:
    if not isinstance(recording_id, str) or not RECORDING_ID_RE.fullmatch(recording_id):
        raise ManualCurationError(
            "recording_id must use only letters, digits, dots, underscores, and dashes "
            "and must begin with a letter or digit."
        )
    return recording_id


def _resolve_target_dir(
    repo_root: Path, recording_id: str, *, output_dir: str | Path | None
) -> Path:
    expected = Path(os.path.abspath(repo_root / "data" / "interim" / "manual_curation" / recording_id))
    if output_dir is None:
        return expected
    raw = Path(output_dir)
    candidate = raw if raw.is_absolute() else repo_root / raw
    candidate = Path(os.path.abspath(candidate))
    if candidate != expected:
        raise ManualCurationError(
            "Curation output must be exactly data/interim/manual_curation/<recording_id>."
        )
    return candidate


def _validate_output_target(
    target_dir: Path,
    repo_root: Path,
    recording_id: str,
    *,
    overwrite: bool,
) -> None:
    expected = Path(os.path.abspath(repo_root / "data" / "interim" / "manual_curation" / recording_id))
    if target_dir != expected:
        raise ManualCurationError("Curation output path is outside the approved manual-curation pool.")

    # Inspect the existing hierarchy through directory descriptors.  A dry
    # run must stay write-free, so a missing parent is acceptable; a present
    # parent must still be real (not a symlink) at every component.
    parent_fd = _open_existing_curation_parent(repo_root)
    if parent_fd is None:
        return
    try:
        target_state = _inspect_directory_entry(parent_fd, recording_id)
        if target_state == "missing":
            return
        if target_state != "directory":
            raise ManualCurationError(
                "Existing curation output target must be a real directory, not a file or symlink."
            )
        if not overwrite:
            raise ManualCurationError(
                "Curation output already exists. Refusing to overwrite it without overwrite=True."
            )
    finally:
        os.close(parent_fd)


def _read_annotation_kit(value: bytes) -> Tuple[List[KitFrame], str | None]:
    kit_bytes = _as_bytes(value, "Annotation kit")
    try:
        archive = zipfile.ZipFile(BytesIO(kit_bytes), "r")
    except zipfile.BadZipFile:
        return [], "Annotation kit is not a valid ZIP archive."

    try:
        infos = archive.infolist()
        if len(infos) > _MAX_KIT_MEMBERS:
            return [], "Annotation kit contains too many archive members."

        names = set()
        for info in infos:
            _validate_zip_info(info)
            if info.filename in names:
                return [], f"Annotation kit has a duplicate archive member: {info.filename}"
            names.add(info.filename)

        if "frame_manifest.csv" not in names:
            return [], "Annotation kit is missing frame_manifest.csv."
        manifest_info = archive.getinfo("frame_manifest.csv")
        if manifest_info.file_size > _MAX_MANIFEST_BYTES:
            return [], "Annotation-kit frame manifest is too large."
        manifest_bytes = archive.read(manifest_info)
        manifest_rows = _parse_kit_manifest(manifest_bytes)

        frames: List[KitFrame] = []
        total_frame_bytes = 0
        seen_indices = set()
        previous_index = -1
        for row in manifest_rows:
            frame_index, timestamp, frame_name, width, height = row
            if frame_index in seen_indices or frame_index <= previous_index:
                return [], "Annotation-kit frame indices must be unique and strictly increasing."
            seen_indices.add(frame_index)
            previous_index = frame_index
            expected_filename = f"frame_{frame_index:05d}.jpg"
            if frame_name != expected_filename or not _FRAME_FILENAME_RE.fullmatch(frame_name):
                return [], "Annotation-kit frame_file must exactly match its frame_index JPEG filename."
            member_name = f"frames/{frame_name}"
            if member_name not in names:
                return [], f"Annotation kit is missing frame image {member_name}."
            frame_info = archive.getinfo(member_name)
            if frame_info.file_size <= 0 or frame_info.file_size > _MAX_FRAME_BYTES:
                return [], f"Annotation-kit frame {frame_name} has an invalid size."
            total_frame_bytes += frame_info.file_size
            if total_frame_bytes > _MAX_TOTAL_FRAME_BYTES:
                return [], "Annotation kit's total frame bytes exceed the safe limit."
            jpeg_bytes = archive.read(frame_info)
            _verify_jpeg(jpeg_bytes, width=width, height=height, frame_name=frame_name)
            frames.append(
                KitFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    frame_file=frame_name,
                    width=width,
                    height=height,
                    jpeg_bytes=jpeg_bytes,
                    sha256=_sha256_bytes(jpeg_bytes),
                )
            )
        return frames, None
    except (KeyError, UnicodeDecodeError, csv.Error, OSError, ValueError, UnidentifiedImageError) as exc:
        return [], f"Unable to read annotation kit safely: {exc}"
    finally:
        archive.close()


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not name or name.startswith(("/", "\\")) or "\\" in name:
        raise ManualCurationError("Annotation kit contains an unsafe ZIP member path.")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        # A trailing slash represents a directory.  It is safe only if it does
        # not contain traversal; the kit does not need directories, so reject it.
        raise ManualCurationError("Annotation kit contains an unsafe ZIP member path.")
    unix_type = (info.external_attr >> 16) & 0o170000
    if unix_type == stat.S_IFLNK:
        raise ManualCurationError("Annotation kit contains a forbidden symbolic-link member.")
    if info.flag_bits & 0x1:
        raise ManualCurationError("Encrypted annotation-kit members are not supported.")
    if info.is_dir():
        raise ManualCurationError("Annotation kit must not contain directory members.")


def _parse_kit_manifest(data: bytes) -> List[Tuple[int, float, str, int, int]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManualCurationError("Annotation-kit frame manifest is not valid UTF-8.") from exc
    try:
        rows = list(csv.reader(StringIO(text), strict=True))
    except csv.Error as exc:
        raise ManualCurationError(f"Annotation-kit frame manifest CSV formatting error: {exc}") from exc
    if not rows:
        raise ManualCurationError("Annotation-kit frame manifest is empty.")
    if rows[0] != _KIT_MANIFEST_HEADERS:
        raise ManualCurationError(
            "Annotation-kit frame manifest headers must exactly be: "
            + ",".join(_KIT_MANIFEST_HEADERS)
        )
    if len(rows) < 2:
        raise ManualCurationError("Annotation-kit frame manifest has no frame rows.")

    parsed: List[Tuple[int, float, str, int, int]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(_KIT_MANIFEST_HEADERS):
            raise ManualCurationError(
                f"Annotation-kit frame manifest line {line_number} must have exactly 5 values."
            )
        raw_index, raw_timestamp, frame_name, raw_width, raw_height = row
        if not INTEGER_RE.fullmatch(raw_index):
            raise ManualCurationError(
                f"Annotation-kit frame manifest line {line_number} has an invalid frame_index."
            )
        if not INTEGER_RE.fullmatch(raw_width) or not INTEGER_RE.fullmatch(raw_height):
            raise ManualCurationError(
                f"Annotation-kit frame manifest line {line_number} has invalid dimensions."
            )
        try:
            timestamp = float(raw_timestamp)
        except ValueError as exc:
            raise ManualCurationError(
                f"Annotation-kit frame manifest line {line_number} has an invalid timestamp."
            ) from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ManualCurationError(
                f"Annotation-kit frame manifest line {line_number} has a non-finite or negative timestamp."
            )
        index = int(raw_index)
        width = int(raw_width)
        height = int(raw_height)
        if width <= 0 or height <= 0:
            raise ManualCurationError(
                f"Annotation-kit frame manifest line {line_number} must have positive dimensions."
            )
        parsed.append((index, timestamp, frame_name, width, height))
    return parsed


def _verify_jpeg(data: bytes, *, width: int, height: int, frame_name: str) -> None:
    if not data:
        raise ManualCurationError(f"Annotation-kit frame {frame_name} is empty.")
    if (
        width > _MAX_FRAME_WIDTH
        or height > _MAX_FRAME_HEIGHT
        or width * height > _MAX_FRAME_PIXELS
    ):
        raise ManualCurationError(
            f"Annotation-kit frame {frame_name} exceeds the safe image-dimension limit."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format != "JPEG":
                    raise ManualCurationError(f"Annotation-kit frame {frame_name} is not a JPEG image.")
                if (
                    image.width > _MAX_FRAME_WIDTH
                    or image.height > _MAX_FRAME_HEIGHT
                    or image.width * image.height > _MAX_FRAME_PIXELS
                ):
                    raise ManualCurationError(
                        f"Annotation-kit frame {frame_name} exceeds the safe image-dimension limit."
                    )
                image.verify()
            with Image.open(BytesIO(data)) as image:
                image.load()
                if image.format != "JPEG" or image.size != (width, height):
                    raise ManualCurationError(
                        f"Annotation-kit frame {frame_name} dimensions do not match its manifest."
                    )
    except ManualCurationError:
        raise
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ManualCurationError(f"Annotation-kit frame {frame_name} is unreadable: {exc}") from exc


def _exportable_jpeg(data: bytes, *, width: int, height: int, frame_name: str) -> bytes:
    """Decode and re-encode a frame without copying source-image metadata."""

    _verify_jpeg(data, width=width, height=height, frame_name=frame_name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                source.load()
                # A new RGB canvas deliberately severs EXIF, XMP, comments,
                # color-profile metadata, and any other source info.
                clean = Image.new("RGB", (width, height))
                clean.paste(source.convert("RGB"))
        encoded = BytesIO()
        clean.save(encoded, format="JPEG", quality=95, optimize=False, progressive=False)
        result = encoded.getvalue()
        _verify_jpeg(result, width=width, height=height, frame_name=frame_name)
        return result
    except ManualCurationError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ManualCurationError(
            f"Annotation-kit frame {frame_name} could not be safely re-encoded."
        ) from exc


def _parse_frame_review_csv(data: bytes, allowed_frames: Sequence[int]) -> Tuple[List[FrameReview], List[str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], ["Frame-review CSV is not valid UTF-8 encoding."]
    try:
        rows = list(csv.reader(StringIO(text), strict=True))
    except csv.Error as exc:
        return [], [f"Frame-review CSV formatting error: {exc}"]
    if not rows:
        return [], ["Frame-review CSV is empty or lacks headers."]
    if rows[0] != _REVIEW_HEADERS:
        return [], [
            "Frame-review CSV headers must exactly be: " + ",".join(_REVIEW_HEADERS)
        ]
    if len(rows) < 2:
        return [], ["Frame-review CSV has no data rows."]

    allowed = set(allowed_frames)
    parsed: List[FrameReview] = []
    seen = set()
    errors: List[str] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        if len(row) != len(_REVIEW_HEADERS):
            errors.append(f"Frame-review CSV line {line_number}: Expected exactly 3 values.")
            continue
        raw_index, status, note = row
        if not INTEGER_RE.fullmatch(raw_index):
            errors.append(f"Frame-review CSV line {line_number}: frame_index must be a non-negative integer.")
            continue
        index = int(raw_index)
        if index not in allowed:
            errors.append(
                f"Frame-review CSV line {line_number}: frame_index {index} is not in the annotation-kit manifest."
            )
            continue
        if index in seen:
            errors.append(f"Frame-review CSV line {line_number}: frame_index {index} appears more than once.")
            continue
        if status not in _REVIEW_STATUSES:
            errors.append(
                f"Frame-review CSV line {line_number}: review_status must be exactly "
                "'pothole_confirmed' or 'no_pothole_confirmed'."
            )
            continue
        seen.add(index)
        parsed.append(FrameReview(frame_index=index, review_status=status, note=note.strip()))

    if errors:
        return [], errors
    if not parsed:
        return [], ["Frame-review CSV has no data rows."]
    return parsed, []


def _is_strict_empty_manual_csv(data: bytes) -> bool:
    """Recognize only the canonical no-box CSV used by the curation exporter.

    This deliberately does not weaken :func:`parse_manual_csv`: malformed CSV,
    alternate headers, or a row containing any non-empty value still fail.
    """

    expected_headers = [
        "incident_id",
        "frame_index",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "label",
        "note",
    ]
    try:
        text = data.decode("utf-8")
        rows = list(csv.reader(StringIO(text), strict=True))
    except (UnicodeDecodeError, csv.Error):
        return False
    if not rows or rows[0] != expected_headers:
        return False
    return all(not row for row in rows[1:])


def _manual_boxes_to_yolo(
    boxes: Iterable[Any], *, width: int, height: int, frame_index: int
) -> str:
    if width <= 0 or height <= 0:
        raise ManualCurationError("Annotation-kit frame dimensions must be positive.")
    lines: List[str] = []
    for box in boxes:
        try:
            x_min = int(box.x_min)
            y_min = int(box.y_min)
            x_max = int(box.x_max)
            y_max = int(box.y_max)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ManualCurationError(
                f"Frame {frame_index} has an invalid manual annotation box."
            ) from exc
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            raise ManualCurationError(
                f"Frame {frame_index} has a manual annotation box outside its pixel dimensions."
            )
        center_x = ((x_min + x_max) / 2.0) / width
        center_y = ((y_min + y_max) / 2.0) / height
        norm_width = (x_max - x_min) / width
        norm_height = (y_max - y_min) / height
        if not all(math.isfinite(value) for value in (center_x, center_y, norm_width, norm_height)):
            raise ManualCurationError(f"Frame {frame_index} has a non-finite converted YOLO box.")
        if not (
            0.0 <= center_x <= 1.0
            and 0.0 <= center_y <= 1.0
            and 0.0 < norm_width <= 1.0
            and 0.0 < norm_height <= 1.0
            and center_x - norm_width / 2.0 >= 0.0
            and center_x + norm_width / 2.0 <= 1.0
            and center_y - norm_height / 2.0 >= 0.0
            and center_y + norm_height / 2.0 <= 1.0
        ):
            raise ManualCurationError(f"Frame {frame_index} has an out-of-bounds converted YOLO box.")
        # The values were validated before normalization.  They are not clamped.
        lines.append(f"0 {center_x:.10f} {center_y:.10f} {norm_width:.10f} {norm_height:.10f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _manifest_rows(recording_id: str, selected: Sequence[Mapping[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for entry in selected:
        frame: KitFrame = entry["frame"]
        review: FrameReview = entry["review"]
        rows.append(
            [
                recording_id,
                frame.frame_index,
                f"{frame.timestamp_seconds:.9f}",
                review.review_status,
                entry["image_relpath"],
                entry["label_relpath"],
                frame.sha256,
                entry["image_sha256"],
                entry["label_sha256"],
                frame.width,
                frame.height,
                entry["box_count"],
            ]
        )
    return rows


def _csv_bytes(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _build_metadata(
    *,
    recording_id: str,
    annotation_kit_bytes: bytes,
    annotations_csv_bytes: bytes,
    frame_review_csv_bytes: bytes,
    selected: Sequence[Mapping[str, Any]],
    manifest_bytes: bytes,
    target_dir: Path,
    repo_root: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    confirmed_positive = sum(
        entry["review"].review_status == "pothole_confirmed" for entry in selected
    )
    confirmed_negative = len(selected) - confirmed_positive
    return {
        "schema_version": 1,
        "mode": "manual_curation_pool",
        "human_reviewed": True,
        "training_or_evaluation_performed": False,
        "recording_id": recording_id,
        "output_relative_path": _repo_relative_posix(target_dir, repo_root),
        "dry_run": dry_run,
        "class_mapping": {"0": "pothole"},
        "source_hashes": {
            "annotation_kit_sha256": _sha256_bytes(annotation_kit_bytes),
            "manual_annotations_csv_sha256": _sha256_bytes(annotations_csv_bytes),
            "frame_review_csv_sha256": _sha256_bytes(frame_review_csv_bytes),
        },
        "counts": {
            "reviewed_frames": len(selected),
            "pothole_confirmed_frames": confirmed_positive,
            "no_pothole_confirmed_frames": confirmed_negative,
            "manual_pothole_boxes": sum(entry["box_count"] for entry in selected),
        },
        "manifest": {
            "path": "manifests/curation_manifest.csv",
            "sha256": _sha256_bytes(manifest_bytes),
        },
        "frame_mappings": [
            {
                "frame_index": entry["frame"].frame_index,
                "timestamp_seconds": entry["frame"].timestamp_seconds,
                "review_status": entry["review"].review_status,
                "image_path": entry["image_relpath"],
                "label_path": entry["label_relpath"],
                "source_image_sha256": entry["frame"].sha256,
                "exported_image_sha256": entry["image_sha256"],
                "label_sha256": entry["label_sha256"],
                "box_count": entry["box_count"],
            }
            for entry in selected
        ],
    }


def _repo_relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ManualCurationError("Curation output path escaped the repository root.") from exc


def _write_batch_atomically(
    *,
    target_dir: Path,
    repo_root: Path,
    selected: Sequence[Mapping[str, Any]],
    manifest_bytes: bytes,
    metadata_bytes: bytes,
    overwrite: bool,
) -> None:
    """Promote a batch using only descriptor-relative output operations.

    Path checks made before this point are valuable diagnostics, but are not a
    complete TOCTOU defence.  Keeping an open directory descriptor for every
    mutation prevents a later symlink swap of ``data/`` or ``data/interim/``
    from redirecting exported user frames outside the approved curation root.
    """

    parent_fd = _open_or_create_curation_parent(repo_root)
    target_name = target_dir.name
    staging_name: str | None = None
    staging_fd: int | None = None
    backup_name: str | None = None
    promoted = False
    try:
        target_state = _inspect_directory_entry(parent_fd, target_name)
        if target_state != "missing" and target_state != "directory":
            raise ManualCurationError(
                "Existing curation output target must be a real directory, not a file or symlink."
            )
        if target_state == "directory" and not overwrite:
            raise ManualCurationError("Curation output already exists; overwrite=True is required.")

        staging_name, staging_fd = _create_private_directory_at(
            parent_fd, prefix=f".{target_name}.stage-"
        )
        _write_batch_contents_at(staging_fd, selected, manifest_bytes, metadata_bytes)
        _assert_complete_staging_at(staging_fd, selected)
        os.close(staging_fd)
        staging_fd = None

        if target_state == "directory":
            backup_name = _unused_child_name(parent_fd, prefix=f".{target_name}.backup-")
            os.rename(
                target_name,
                backup_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        try:
            os.rename(
                staging_name,
                target_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            promoted = True
        except OSError as exc:
            if backup_name is not None:
                try:
                    os.rename(
                        backup_name,
                        target_name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                except OSError as rollback_exc:
                    raise ManualCurationError(
                        "Unable to promote curation output atomically; the prior batch was retained at "
                        f"data/interim/manual_curation/{backup_name} because rollback failed: {rollback_exc}"
                    ) from exc
            raise ManualCurationError(f"Unable to promote curation output atomically: {exc}") from exc

        target_fd = _open_directory_at(parent_fd, target_name)
        final_verification_error: BaseException | None = None
        try:
            _assert_complete_staging_at(target_fd, selected)
        except BaseException as exc:
            final_verification_error = exc
        finally:
            os.close(target_fd)
        if final_verification_error is not None:
            rollback_error = _rollback_promoted_output(
                parent_fd=parent_fd,
                target_name=target_name,
                backup_name=backup_name,
            )
            if rollback_error is not None:
                raise ManualCurationError(
                    "Promoted curation output did not pass final verification and rollback failed: "
                    + rollback_error
                ) from final_verification_error
            message = (
                "Promoted curation output did not pass final verification; the prior batch was restored."
                if backup_name is not None
                else "Promoted curation output did not pass final verification; the new batch was removed."
            )
            raise ManualCurationError(message) from final_verification_error

        if backup_name is not None:
            _remove_owned_directory_at(parent_fd, backup_name)
    except BaseException:
        if staging_fd is not None:
            os.close(staging_fd)
        if not promoted and staging_name is not None:
            try:
                _remove_owned_directory_at(parent_fd, staging_name)
            except ManualCurationError:
                # Preserve an unsafe or unexpectedly modified staging tree for
                # inspection rather than recursing through it.
                pass
        raise
    finally:
        os.close(parent_fd)


def _open_existing_curation_parent(repo_root: Path) -> int | None:
    """Return a safe descriptor for the existing curation parent, if present.

    This is used by dry-run validation and intentionally creates nothing.
    """

    current_fd = _open_real_directory_path(repo_root, "Repository root")
    try:
        for component in ("data", "interim", "manual_curation"):
            try:
                next_fd = _open_directory_at(current_fd, component)
            except FileNotFoundError:
                os.close(current_fd)
                return None
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        try:
            os.close(current_fd)
        except OSError:
            pass
        raise


def _open_or_create_curation_parent(repo_root: Path) -> int:
    """Create/open only fixed child directories using ``openat`` semantics."""

    current_fd = _open_real_directory_path(repo_root, "Repository root")
    try:
        for component in ("data", "interim", "manual_curation"):
            try:
                next_fd = _open_directory_at(current_fd, component)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                except FileExistsError:
                    # Another local process may have created it; immediately
                    # reopen it with O_NOFOLLOW before trusting it.
                    pass
                next_fd = _open_directory_at(current_fd, component)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        try:
            os.close(current_fd)
        except OSError:
            pass
        raise


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_real_directory_path(path: Path, label: str) -> int:
    try:
        file_fd = os.open(path, _directory_open_flags())
    except OSError as exc:
        raise ManualCurationError(f"{label} is unavailable or unsafe: {exc}") from exc
    try:
        if not stat.S_ISDIR(os.fstat(file_fd).st_mode):
            raise ManualCurationError(f"{label} must be a real directory, not a symlink.")
        return file_fd
    except BaseException:
        os.close(file_fd)
        raise


def _open_directory_at(parent_fd: int, name: str) -> int:
    file_fd = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    try:
        if not stat.S_ISDIR(os.fstat(file_fd).st_mode):
            raise ManualCurationError("Curation output path contains a non-directory component.")
        return file_fd
    except BaseException:
        os.close(file_fd)
        raise


def _inspect_directory_entry(parent_fd: int, name: str) -> str:
    try:
        info = os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        raise ManualCurationError(f"Unable to inspect curation output target safely: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    return "other"


def _create_private_directory_at(parent_fd: int, *, prefix: str) -> Tuple[str, int]:
    for _ in range(32):
        name = f"{prefix}{uuid.uuid4().hex}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            file_fd = _open_directory_at(parent_fd, name)
            os.fchmod(file_fd, 0o700)
            return name, file_fd
        except BaseException:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    raise ManualCurationError("Unable to reserve a unique curation staging directory.")


def _unused_child_name(parent_fd: int, *, prefix: str) -> str:
    for _ in range(32):
        name = f"{prefix}{uuid.uuid4().hex}"
        if _inspect_directory_entry(parent_fd, name) == "missing":
            return name
    raise ManualCurationError("Unable to reserve a unique curation backup directory name.")


def _write_batch_contents_at(
    staging_fd: int,
    selected: Sequence[Mapping[str, Any]],
    manifest_bytes: bytes,
    metadata_bytes: bytes,
) -> None:
    images_fd: int | None = None
    labels_fd: int | None = None
    manifests_fd: int | None = None
    # Rename fixed private directories only after opening descriptors for them;
    # this guarantees names and types cannot be redirected through a symlink.
    try:
        images_name, images_fd = _create_private_directory_at(staging_fd, prefix="images-")
        labels_name, labels_fd = _create_private_directory_at(staging_fd, prefix="labels-")
        manifests_name, manifests_fd = _create_private_directory_at(staging_fd, prefix="manifests-")
        os.rename(images_name, "images", src_dir_fd=staging_fd, dst_dir_fd=staging_fd)
        os.rename(labels_name, "labels", src_dir_fd=staging_fd, dst_dir_fd=staging_fd)
        os.rename(manifests_name, "manifests", src_dir_fd=staging_fd, dst_dir_fd=staging_fd)
        for entry in selected:
            frame: KitFrame = entry["frame"]
            _write_private_file_at(images_fd, frame.frame_file, entry["image_bytes"])
            _write_private_file_at(
                labels_fd,
                Path(frame.frame_file).with_suffix(".txt").name,
                entry["label_bytes"],
            )

        _write_private_file_at(manifests_fd, "curation_manifest.csv", manifest_bytes)
        _write_private_file_at(manifests_fd, "curation_metadata.json", metadata_bytes)
    finally:
        for file_fd in (images_fd, labels_fd, manifests_fd):
            if file_fd is not None:
                os.close(file_fd)


def _write_private_file_at(parent_fd: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    file_fd: int | None = None
    try:
        file_fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
        os.fchmod(file_fd, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError("Unable to write complete curation output file.")
            view = view[written:]
        os.fsync(file_fd)
    except OSError as exc:
        raise ManualCurationError(f"Unable to write curation output file safely: {exc}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _assert_complete_staging_at(staging_fd: int, selected: Sequence[Mapping[str, Any]]) -> None:
    images_fd = _open_directory_at(staging_fd, "images")
    labels_fd = _open_directory_at(staging_fd, "labels")
    manifests_fd = _open_directory_at(staging_fd, "manifests")
    try:
        _assert_complete_staging_contents_at(images_fd, labels_fd, manifests_fd, selected)
    finally:
        os.close(images_fd)
        os.close(labels_fd)
        os.close(manifests_fd)


def _assert_complete_staging_contents_at(
    images_fd: int,
    labels_fd: int,
    manifests_fd: int,
    selected: Sequence[Mapping[str, Any]],
) -> None:
    for entry in selected:
        frame: KitFrame = entry["frame"]
        label_name = Path(frame.frame_file).with_suffix(".txt").name
        if not _is_regular_file_at(images_fd, frame.frame_file) or _sha256_file_at(
            images_fd, frame.frame_file
        ) != entry["image_sha256"]:
            raise ManualCurationError("Staged curation image verification failed.")
        if (
            not _is_regular_file_at(labels_fd, label_name)
            or _sha256_file_at(labels_fd, label_name) != entry["label_sha256"]
        ):
            raise ManualCurationError("Staged curation label verification failed.")
    for name in ("curation_manifest.csv", "curation_metadata.json"):
        if not _is_regular_file_at(manifests_fd, name) or _stat_at(manifests_fd, name).st_size <= 0:
            raise ManualCurationError("Staged curation manifest verification failed.")


def _rollback_promoted_output(
    *, parent_fd: int, target_name: str, backup_name: str | None
) -> str | None:
    """Remove the just-promoted batch and restore a backup where possible."""

    try:
        _remove_owned_directory_at(parent_fd, target_name)
    except ManualCurationError as exc:
        return f"the new output could not be safely removed: {exc}"
    if backup_name is None:
        return None
    try:
        os.rename(
            backup_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        return None
    except OSError as exc:
        return (
            "the prior batch was retained at "
            f"data/interim/manual_curation/{backup_name}: {exc}"
        )


def _remove_owned_directory_at(parent_fd: int, name: str) -> None:
    """Remove only a descriptor-confined temporary/backup directory tree."""

    state = _inspect_directory_entry(parent_fd, name)
    if state == "missing":
        return
    if state != "directory":
        raise ManualCurationError("Temporary curation directory became unsafe; it was retained.")
    directory_fd = _open_directory_at(parent_fd, name)
    try:
        _remove_directory_contents_at(directory_fd)
    finally:
        os.close(directory_fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except OSError as exc:
        raise ManualCurationError(f"Unable to remove temporary curation directory: {exc}") from exc


def _remove_directory_contents_at(directory_fd: int) -> None:
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise ManualCurationError(f"Unable to inspect temporary curation directory: {exc}") from exc
    for name in names:
        try:
            info = os.lstat(name, dir_fd=directory_fd)
        except OSError as exc:
            raise ManualCurationError(f"Unable to inspect temporary curation entry: {exc}") from exc
        if stat.S_ISDIR(info.st_mode):
            _remove_owned_directory_at(directory_fd, name)
        else:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except OSError as exc:
                raise ManualCurationError(f"Unable to remove temporary curation file: {exc}") from exc


def _stat_at(parent_fd: int, name: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ManualCurationError(f"Unable to inspect curation output file: {exc}") from exc


def _is_regular_file_at(parent_fd: int, name: str) -> bool:
    try:
        return stat.S_ISREG(_stat_at(parent_fd, name).st_mode)
    except ManualCurationError:
        return False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_at(parent_fd: int, name: str) -> str:
    digest = hashlib.sha256()
    file_fd: int | None = None
    try:
        file_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ManualCurationError("Curation output entry is not a regular file.")
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    except OSError as exc:
        raise ManualCurationError(f"Unable to hash staged curation output: {exc}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
    return digest.hexdigest()
