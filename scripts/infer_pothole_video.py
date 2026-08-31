"""Run the pinned local frozen model on a user video as experimental suggestions.

This command never trains, validates, downloads weights, or reads the prepared
training dataset.  It is intentionally separate from the one-time held-out
test evaluation command.
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import hashlib
import json
import re
import stat
import zipfile
from io import BytesIO

from src.config import load_config
from src.ml.experimental_inference import (
    ExperimentalInferenceSettings,
    run_verified_sampled_video_inference,
)
from src.ml.local_model_runtime import (
    LocalModelRuntimeError,
    load_verified_yolo_model,
    validate_requested_device,
)
from src.ml.model_provenance import (
    FrozenBaselineVerificationError,
    load_frozen_baseline_config,
    verify_frozen_baseline,
)
from src.video_io import process_and_create_kit, validate_video


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _exit(message: str) -> None:
    print(f"Error: {message}")
    raise SystemExit(1)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ensure_regular_input(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    try:
        info = os.lstat(path)
    except OSError as exc:
        _exit(f"Input video is missing or unreadable: {exc}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _exit("Input video must be a regular local file, not a symlink or directory.")
    return path.resolve()


def _assert_no_symlink_components(path: Path) -> None:
    repo_root = REPO_ROOT.resolve()
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        _exit("Output path must be inside the repository root.")

    current = repo_root
    try:
        root_info = os.lstat(current)
    except OSError as exc:
        _exit(f"Repository root is inaccessible: {exc}")
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        _exit("Repository root must be a real directory.")

    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            # Descendants cannot exist once an ancestor does not exist.
            break
        except OSError as exc:
            _exit(f"Unable to inspect output path component {current}: {exc}")
        if stat.S_ISLNK(info.st_mode):
            _exit(f"Output path contains a forbidden symlink component: {current}")


def _resolve_output_dir(output_value: str) -> Path:
    if not isinstance(output_value, str) or not output_value or output_value != output_value.strip():
        _exit("--output-dir must be a non-empty, whitespace-trimmed path.")

    raw = Path(output_value)
    candidate = raw if raw.is_absolute() else REPO_ROOT / raw
    candidate = Path(os.path.abspath(candidate))
    output_root = REPO_ROOT.resolve() / "outputs" / "inference"
    try:
        relative = candidate.relative_to(output_root)
    except ValueError:
        _exit("--output-dir must be a direct child of outputs/inference/.")
    if len(relative.parts) != 1 or not RUN_ID_RE.fullmatch(relative.name):
        _exit("--output-dir must name one safe new child directory under outputs/inference/.")

    _assert_no_symlink_components(candidate)
    if candidate.exists() or candidate.is_symlink():
        _exit("--output-dir already exists; experimental output is never overwritten.")
    return candidate


def _frame_indices_from_kit(kit_zip: bytes) -> list[int]:
    expected = ["frame_index", "timestamp_seconds", "frame_file", "width", "height"]
    try:
        with zipfile.ZipFile(BytesIO(kit_zip)) as archive:
            reader = csv.DictReader(archive.read("frame_manifest.csv").decode("utf-8").splitlines())
            if reader.fieldnames != expected:
                _exit("The generated annotation-kit frame manifest has unexpected columns.")
            archive_names = set(archive.namelist())
            indices = []
            previous = -1
            for row in reader:
                index = int(row["frame_index"])
                frame_name = row["frame_file"]
                if index < 0 or index <= previous:
                    _exit("The generated annotation-kit frame manifest has invalid frame ordering.")
                if not frame_name.startswith("frame_") or "/" in frame_name or "\\" in frame_name:
                    _exit("The generated annotation-kit frame manifest has an unsafe frame filename.")
                if f"frames/{frame_name}" not in archive_names:
                    _exit("A frame listed in the annotation-kit frame manifest is missing.")
                indices.append(index)
                previous = index
    except (KeyError, UnicodeDecodeError, csv.Error, zipfile.BadZipFile, ValueError) as exc:
        _exit(f"Unable to read the generated annotation-kit frame manifest: {exc}")
    if not indices:
        _exit("The generated annotation kit contains no sampled frames.")
    return indices


def _no_follow_flag() -> int:
    """Return the platform's no-follow flag when the operating system supplies it."""

    return getattr(os, "O_NOFOLLOW", 0)


def _open_safe_directory(path: Path, *, label: str) -> int:
    """Open one already-checked directory and bind its pathname to an inode."""

    try:
        expected = os.lstat(path)
    except OSError as exc:
        _exit(f"Unable to inspect {label}: {exc}")
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
        _exit(f"{label} must be a real directory, not a symlink.")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _no_follow_flag()
    try:
        directory_fd = os.open(path, flags)
    except OSError as exc:
        _exit(f"Unable to open {label} safely: {exc}")

    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        os.close(directory_fd)
        _exit(f"{label} changed while it was being opened; refusing to write output.")
    return directory_fd


def _write_all(file_fd: int, data: bytes) -> None:
    """Write bytes through an already-safe descriptor without reopening a path."""

    view = memoryview(data)
    while view:
        written = os.write(file_fd, view)
        if written <= 0:
            raise OSError("unable to write a complete private output file")
        view = view[written:]


def _write_private_file_atomic(directory_fd: int, final_name: str, data: bytes) -> None:
    """Create one 0600 output file with descriptor-relative, no-overwrite promotion."""

    if not final_name or "/" in final_name or "\\" in final_name:
        raise ValueError("Internal output filename is unsafe.")
    temporary_name = f".{final_name}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag()
    file_fd: int | None = None
    try:
        file_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(file_fd, 0o600)
        _write_all(file_fd, data)
        os.fsync(file_fd)
    finally:
        if file_fd is not None:
            os.close(file_fd)

    try:
        # ``link`` refuses to overwrite an existing destination, unlike rename.
        # Both names are resolved relative to the already-verified private folder.
        os.link(
            temporary_name,
            final_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise


def _remove_owned_output_files(directory_fd: int) -> None:
    """Best-effort cleanup limited to names this command itself can create."""

    for name in (
        ".experimental_inference_report.zip.tmp",
        "experimental_inference_report.zip",
        ".run_summary.json.tmp",
        "run_summary.json",
    ):
        try:
            os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError:
            # Never recursively follow or remove unexpected content on cleanup.
            pass


def _write_new_output(target: Path, report_zip: bytes, summary: dict) -> Path:
    """Create a fresh private output directory only after inference succeeds.

    The directory is created at mode 0700.  Result files are made at mode 0600
    through no-follow, descriptor-relative operations, so annotated user frames
    are not left readable by other local accounts.
    """

    output_root = target.parent
    _assert_no_symlink_components(output_root)
    if not output_root.exists():
        try:
            output_root.mkdir(parents=False, mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            _exit(f"Unable to create outputs/inference directory: {exc}")
    _assert_no_symlink_components(output_root)

    output_root_fd = _open_safe_directory(output_root, label="outputs/inference")
    target_fd: int | None = None
    created_target = False
    created_target_info: os.stat_result | None = None
    try:
        try:
            os.mkdir(target.name, mode=0o700, dir_fd=output_root_fd)
            created_target = True
            created_target_info = os.stat(target.name, dir_fd=output_root_fd, follow_symlinks=False)
        except FileExistsError:
            _exit("--output-dir was created while inference was running; refusing to overwrite it.")
        except OSError as exc:
            _exit(f"Unable to reserve the requested output directory: {exc}")

        target_fd = os.open(
            target.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _no_follow_flag(),
            dir_fd=output_root_fd,
        )
        target_info = os.fstat(target_fd)
        if (
            created_target_info is None
            or not stat.S_ISDIR(target_info.st_mode)
            or (target_info.st_dev, target_info.st_ino)
            != (created_target_info.st_dev, created_target_info.st_ino)
        ):
            _exit("Reserved output directory became unsafe.")
        os.fchmod(target_fd, 0o700)

        _write_private_file_atomic(target_fd, "experimental_inference_report.zip", report_zip)
        report_info = os.stat("experimental_inference_report.zip", dir_fd=target_fd, follow_symlinks=False)
        if not stat.S_ISREG(report_info.st_mode) or report_info.st_size <= 0:
            _exit("Experimental inference report is empty.")

        summary_bytes = json.dumps(summary, indent=2, sort_keys=True).encode("utf-8")
        _write_private_file_atomic(target_fd, "run_summary.json", summary_bytes)
        return target / "experimental_inference_report.zip"
    except BaseException:
        if target_fd is not None:
            _remove_owned_output_files(target_fd)
            os.close(target_fd)
            target_fd = None
        if created_target:
            try:
                os.rmdir(target.name, dir_fd=output_root_fd)
            except OSError:
                # Retain an unexpected directory instead of recursively deleting
                # content that could have been inserted by another process.
                pass
        raise
    finally:
        if target_fd is not None:
            os.close(target_fd)
        os.close(output_root_fd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pinned local frozen pothole model on a video as experimental suggestions."
    )
    parser.add_argument("--input-video", required=True, help="Path to a local road video. It is read only.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New direct child of outputs/inference/ for the experimental ZIP report.",
    )
    parser.add_argument(
        "--frozen-baseline-config",
        default="configs/inference/frozen_baseline.yaml",
        help="Pinned local frozen-baseline provenance config.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional review-display threshold (not a calibrated risk score).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify local model/video/sampling/output safety without loading YOLO or writing output.",
    )
    args = parser.parse_args()

    app_config = load_config(str(REPO_ROOT / "configs" / "default.yaml"))
    if not app_config.experimental_inference.enabled:
        _exit("Experimental inference is disabled in configs/default.yaml.")

    input_path = _ensure_regular_input(args.input_video)
    try:
        input_bytes = input_path.read_bytes()
    except OSError as exc:
        _exit(f"Unable to read input video: {exc}")

    valid, message = validate_video(input_path.name, len(input_bytes), app_config)
    if not valid:
        _exit(message)

    output_dir = _resolve_output_dir(args.output_dir)
    try:
        frozen_config = load_frozen_baseline_config(
            args.frozen_baseline_config, repo_root=REPO_ROOT
        )
        model_info = verify_frozen_baseline(frozen_config, repo_root=REPO_ROOT)
        metadata, kit_zip = process_and_create_kit(input_bytes, input_path.name, app_config)
        sampled_indices = _frame_indices_from_kit(kit_zip)
    except (FrozenBaselineVerificationError, ValueError, OSError) as exc:
        _exit(str(exc))

    threshold = (
        app_config.experimental_inference.default_confidence_threshold
        if args.confidence_threshold is None
        else args.confidence_threshold
    )
    try:
        settings = ExperimentalInferenceSettings(
            device=app_config.experimental_inference.device,
            image_size=app_config.experimental_inference.image_size,
            confidence_threshold=threshold,
            iou_threshold=app_config.experimental_inference.iou_threshold,
            max_detections_per_frame=app_config.experimental_inference.max_detections_per_frame,
            output_fps=app_config.experimental_inference.output_fps,
        )
    except ValueError as exc:
        _exit(str(exc))

    input_sha256 = _sha256_bytes(input_bytes)
    if args.dry_run:
        print("Dry run passed. No model was loaded and no output was written.")
        print(f"Input video: {input_path.name}")
        print(f"Input SHA-256: {input_sha256}")
        print(f"Pinned model SHA-256: {model_info.checkpoint_sha256}")
        print(f"Sampled frames: {len(sampled_indices)}")
        print(f"Configured device (not initialized): {settings.device}")
        return

    try:
        device = validate_requested_device(settings.device)
        settings = ExperimentalInferenceSettings(
            device=device,
            image_size=settings.image_size,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            max_detections_per_frame=settings.max_detections_per_frame,
            output_fps=settings.output_fps,
        )
        model = load_verified_yolo_model(model_info.checkpoint_path, model_info.checkpoint_sha256)
        result = run_verified_sampled_video_inference(
            video_bytes=input_bytes,
            video_filename=input_path.name,
            sampled_frame_indices=sampled_indices,
            model=model,
            model_info=model_info,
            settings=settings,
            input_video_sha256=input_sha256,
        )
    except (LocalModelRuntimeError, RuntimeError, ValueError, OSError) as exc:
        _exit(str(exc))

    report_path = _write_new_output(
        output_dir,
        result.report_zip,
        {
            "status": "experimental_local_model_suggestions_complete",
            "input_video_sha256": input_sha256,
            "baseline_run_id": model_info.run_id,
            "checkpoint_sha256": model_info.checkpoint_sha256,
            "total_sampled_frames": result.total_sampled_frames,
            "frames_with_detections": result.frames_with_detections,
            "total_raw_model_detections": len(result.detections),
            "human_verification_status": "not_human_verified",
        },
    )
    print("Experimental local model suggestions completed.")
    print(f"Report ZIP: {report_path.relative_to(REPO_ROOT)}")
    print(f"Raw unverified detections: {len(result.detections)}")
    print("No training, held-out test evaluation, or manual incident report was performed.")


if __name__ == "__main__":
    main()
