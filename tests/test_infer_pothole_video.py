import importlib.util
import os
import stat
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import AppConfig, Config, ExperimentalInferenceConfig, VideoConfig


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "infer_pothole_video.py"
    spec = importlib.util.spec_from_file_location("infer_pothole_video_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_target_is_a_new_direct_child_and_does_not_create_anything(tmp_path, monkeypatch):
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    (repo_root / "outputs").mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)

    target = module._resolve_output_dir("outputs/inference/demo-run")
    assert target == repo_root / "outputs" / "inference" / "demo-run"
    assert not (repo_root / "outputs" / "inference").exists()

    with pytest.raises(SystemExit):
        module._resolve_output_dir("outputs/inference")
    with pytest.raises(SystemExit):
        module._resolve_output_dir("outputs/inference/demo-run/nested")

    target.parent.mkdir()
    target.mkdir()
    with pytest.raises(SystemExit):
        module._resolve_output_dir("outputs/inference/demo-run")


def test_script_keeps_ml_framework_loading_lazy():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "infer_pothole_video.py"
    source = script_path.read_text(encoding="utf-8")
    assert "from ultralytics" not in source
    assert "import ultralytics" not in source
    assert "import torch" not in source


def test_write_new_output_uses_private_directory_and_files(tmp_path, monkeypatch):
    """Annotated-video reports must not be readable by other local accounts."""

    module = _load_script_module()
    repo_root = tmp_path / "repo"
    (repo_root / "outputs").mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)

    target = repo_root / "outputs" / "inference" / "private-run"
    report = b"not-a-real-zip-but-a-nonempty-private-report"
    summary = {"status": "complete", "human_verification_status": "not_human_verified"}

    report_path = module._write_new_output(target, report, summary)

    assert report_path == target / "experimental_inference_report.zip"
    assert report_path.read_bytes() == report
    assert (target / "run_summary.json").is_file()
    for path in (target.parent, target):
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    for path in (report_path, target / "run_summary.json"):
        info = os.lstat(path)
        assert stat.S_ISREG(info.st_mode)
        assert stat.S_IMODE(info.st_mode) & 0o077 == 0


def test_private_file_writer_never_overwrites_a_colliding_temp_name(tmp_path, monkeypatch):
    """A pre-created temp file must cause failure, never a destructive overwrite."""

    module = _load_script_module()
    repo_root = tmp_path / "repo"
    target = repo_root / "outputs" / "inference" / "private-run"
    target.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)

    temp_path = target / ".experimental_inference_report.zip.tmp"
    temp_path.write_bytes(b"sentinel content")
    directory_fd = module._open_safe_directory(target, label="test output")
    try:
        with pytest.raises(FileExistsError):
            module._write_private_file_atomic(
                directory_fd, "experimental_inference_report.zip", b"new report"
            )
    finally:
        os.close(directory_fd)

    assert temp_path.read_bytes() == b"sentinel content"
    assert not (target / "experimental_inference_report.zip").exists()


def test_private_file_writer_does_not_follow_or_replace_a_final_name_symlink(tmp_path, monkeypatch):
    """A hostile final-name symlink remains untouched instead of being followed."""

    module = _load_script_module()
    repo_root = tmp_path / "repo"
    target = repo_root / "outputs" / "inference" / "private-run"
    target.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)

    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"do not replace")
    final_path = target / "experimental_inference_report.zip"
    final_path.symlink_to(outside)

    directory_fd = module._open_safe_directory(target, label="test output")
    try:
        with pytest.raises(FileExistsError):
            module._write_private_file_atomic(directory_fd, final_path.name, b"new report")
    finally:
        os.close(directory_fd)

    assert final_path.is_symlink()
    assert outside.read_bytes() == b"do not replace"
    assert not (target / ".experimental_inference_report.zip.tmp").exists()


def _write_synthetic_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    assert writer.isOpened(), "OpenCV could not create the synthetic MP4 test fixture."
    for index in range(3):
        frame = np.full((32, 32, 3), index * 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_dry_run_reads_a_synthetic_video_without_loading_model_or_writing_output(
    tmp_path, monkeypatch, capsys
):
    """The real CLI dry-run path must stay model-free and write-free."""

    module = _load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)

    config = Config(
        app=AppConfig(name="Test", max_upload_mb=10),
        video=VideoConfig(allowed_extensions=[".mp4"], sampling_fps=1, max_sampled_frames=2),
        experimental_inference=ExperimentalInferenceConfig(enabled=True, device="mps"),
    )
    input_video = tmp_path / "synthetic.mp4"
    _write_synthetic_video(input_video)

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(module, "load_frozen_baseline_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module,
        "verify_frozen_baseline",
        lambda *_args, **_kwargs: SimpleNamespace(checkpoint_sha256="a" * 64),
    )

    def unexpected_model_load(*_args, **_kwargs):
        raise AssertionError("--dry-run must not load YOLO.")

    def unexpected_report_write(*_args, **_kwargs):
        raise AssertionError("--dry-run must not create an output report.")

    monkeypatch.setattr(module, "load_verified_yolo_model", unexpected_model_load)
    monkeypatch.setattr(module, "validate_requested_device", unexpected_model_load)
    monkeypatch.setattr(module, "_write_new_output", unexpected_report_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_pothole_video.py",
            "--input-video",
            str(input_video),
            "--output-dir",
            "outputs/inference/dry-run-example",
            "--dry-run",
        ],
    )

    module.main()

    stdout = capsys.readouterr().out
    assert "Dry run passed. No model was loaded and no output was written." in stdout
    assert "Sampled frames: 2" in stdout
    assert not (repo_root / "outputs" / "inference").exists()
