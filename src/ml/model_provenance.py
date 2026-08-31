"""Fail-closed provenance checks for the frozen local pothole baseline.

This module deliberately performs no model loading and has no Torch or
Ultralytics imports.  It only verifies local regular files and their pinned
provenance before a caller is allowed to construct an inference model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ARTIFACT_KEYS = (
    "base_weights",
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "train_config.yaml",
    "dataset_preparation_metadata.json",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXPECTED_CLASS_MAPPING = {"0": "pothole"}


class FrozenBaselineVerificationError(RuntimeError):
    """Raised when the pinned frozen baseline cannot be verified safely."""


@dataclass(frozen=True)
class FrozenBaselineConfig:
    """Validated, repository-local pins read from ``frozen_baseline.yaml``."""

    config_path: Path
    repo_root: Path
    run_id: str
    run_directory: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    model_metadata_path: Path
    model_metadata_sha256: str
    base_weights_path: Path
    task: str
    class_mapping: Mapping[str, str]
    git_sha: str
    dataset_fingerprint: str
    artifact_hashes: Mapping[str, str]


@dataclass(frozen=True)
class FrozenModelInfo:
    """Verified device-independent provenance returned to inference callers."""

    run_id: str
    training_run_directory: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    model_metadata_path: Path
    model_metadata_sha256: str
    base_weights_path: Path
    base_weights_sha256: str
    git_sha: str
    dataset_fingerprint: str
    task: str
    class_mapping: Mapping[str, str]
    artifact_hashes: Mapping[str, str]


def _fail(message: str) -> None:
    raise FrozenBaselineVerificationError(message)


def _normalise_repo_root(repo_root: Path | str) -> Path:
    root = Path(os.path.abspath(os.fspath(repo_root)))
    try:
        info = os.lstat(root)
    except OSError as exc:
        _fail(f"Repository root is not accessible: {root} ({exc})")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _fail(f"Repository root must be a real, non-symlink directory: {root}")
    return root


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _local_path(value: object, repo_root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{field} must be a non-empty, whitespace-trimmed path string.")

    raw_path = Path(value)
    candidate = raw_path if raw_path.is_absolute() else repo_root / raw_path
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    if not _is_within(candidate, repo_root):
        _fail(f"{field} must stay inside the repository root: {value!r}")
    return candidate


def _ensure_non_symlink_chain(path: Path, repo_root: Path, field: str) -> None:
    """Ensure every component from the repository root to ``path`` is real."""

    if not _is_within(path, repo_root):
        _fail(f"{field} is outside the repository root: {path}")

    try:
        root_info = os.lstat(repo_root)
    except OSError as exc:
        _fail(f"Unable to inspect repository root for {field}: {exc}")
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        _fail(f"Repository root is unsafe while checking {field}: {repo_root}")

    current = repo_root
    parts = path.relative_to(repo_root).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as exc:
            _fail(f"{field} component is missing or unreadable: {current} ({exc})")
        if stat.S_ISLNK(info.st_mode):
            _fail(f"{field} contains a forbidden symlink component: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            _fail(f"{field} contains a non-directory path component: {current}")


def _ensure_safe_regular_file(path: Path, repo_root: Path, field: str) -> None:
    _ensure_non_symlink_chain(path, repo_root, field)
    try:
        info = os.lstat(path)
    except OSError as exc:
        _fail(f"{field} is missing or unreadable: {path} ({exc})")
    if not stat.S_ISREG(info.st_mode):
        _fail(f"{field} must be a regular, non-symlink file: {path}")


def _ensure_safe_directory(path: Path, repo_root: Path, field: str) -> None:
    _ensure_non_symlink_chain(path, repo_root, field)
    try:
        info = os.lstat(path)
    except OSError as exc:
        _fail(f"{field} is missing or unreadable: {path} ({exc})")
    if not stat.S_ISDIR(info.st_mode):
        _fail(f"{field} must be a real, non-symlink directory: {path}")


def _sha256(path: Path, repo_root: Path, field: str) -> str:
    _ensure_safe_regular_file(path, repo_root, field)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"Unable to read {field} safely: {path} ({exc})")
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(f"{field} must be a lowercase 64-character SHA-256 digest.")
    return value


def _normalise_class_mapping(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping.")

    normalised: dict[str, str] = {}
    for raw_key, raw_label in value.items():
        if isinstance(raw_key, bool):
            _fail(f"{field} has an invalid boolean class id.")
        if isinstance(raw_key, int):
            key = str(raw_key)
        elif isinstance(raw_key, str) and re.fullmatch(r"[0-9]+", raw_key):
            key = raw_key
        else:
            _fail(f"{field} has an invalid class id: {raw_key!r}")
        if not isinstance(raw_label, str) or raw_label != raw_label.strip() or not raw_label:
            _fail(f"{field}[{key!r}] must be a non-empty, whitespace-trimmed label.")
        if key in normalised:
            _fail(f"{field} repeats class id {key!r}.")
        normalised[key] = raw_label
    return normalised


def _read_yaml_mapping(path: Path, repo_root: Path, field: str) -> dict[str, Any]:
    _ensure_safe_regular_file(path, repo_root, field)
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        _fail(f"Unable to parse {field}: {exc}")
    if not isinstance(parsed, dict):
        _fail(f"{field} must contain a YAML mapping.")
    return parsed


def _read_json_mapping(path: Path, repo_root: Path, field: str) -> dict[str, Any]:
    _ensure_safe_regular_file(path, repo_root, field)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"Unable to parse {field}: {exc}")
    if not isinstance(parsed, dict):
        _fail(f"{field} must contain a JSON object.")
    return parsed


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        _fail(f"{field} has an invalid schema ({'; '.join(details)}).")


def _expected_artifact_path(key: str, config: FrozenBaselineConfig) -> Path:
    if key == "base_weights":
        return config.base_weights_path
    return config.run_directory / key


def _strict_metadata_absolute_path(value: object, expected: Path, field: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{field} must be a non-empty, whitespace-trimmed path string.")
    raw = Path(value)
    if not raw.is_absolute() or ".." in raw.parts:
        _fail(f"{field} must be an absolute canonical local path.")
    normalised = Path(os.path.abspath(os.fspath(raw)))
    if normalised != expected:
        _fail(f"{field} does not match the pinned local path ({normalised} != {expected}).")
    return normalised


def load_frozen_baseline_config(
    config_path: Path | str,
    repo_root: Path | str = REPO_ROOT,
) -> FrozenBaselineConfig:
    """Load a strict frozen-baseline configuration without opening model weights."""

    root = _normalise_repo_root(repo_root)
    if not isinstance(config_path, (str, os.PathLike)):
        _fail("frozen baseline config path must be a path string or PathLike value.")
    path = _local_path(os.fspath(config_path), root, "frozen baseline config path")
    payload = _read_yaml_mapping(path, root, "frozen baseline config")
    _require_exact_keys(payload, {"schema_version", "frozen_baseline"}, "frozen baseline config")

    if payload["schema_version"] != 1:
        _fail("frozen baseline config must declare schema_version: 1.")
    baseline = payload["frozen_baseline"]
    if not isinstance(baseline, dict):
        _fail("frozen_baseline must be a YAML mapping.")

    expected_keys = {
        "run_id",
        "run_directory",
        "checkpoint_path",
        "checkpoint_sha256",
        "model_metadata_path",
        "model_metadata_sha256",
        "base_weights_path",
        "task",
        "class_mapping",
        "git_sha",
        "dataset_fingerprint",
        "artifact_hashes",
    }
    _require_exact_keys(baseline, expected_keys, "frozen_baseline")

    run_id = baseline["run_id"]
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        _fail("frozen_baseline.run_id must be a safe local run identifier.")

    expected_run_directory = root / "outputs" / "training" / run_id
    run_directory = _local_path(baseline["run_directory"], root, "frozen_baseline.run_directory")
    if run_directory != expected_run_directory:
        _fail("frozen_baseline.run_directory must be exactly outputs/training/<run_id>.")

    expected_checkpoint = expected_run_directory / "weights" / "best.pt"
    checkpoint_path = _local_path(baseline["checkpoint_path"], root, "frozen_baseline.checkpoint_path")
    if checkpoint_path != expected_checkpoint:
        _fail("frozen_baseline.checkpoint_path must be exactly <run>/weights/best.pt.")

    expected_metadata = expected_run_directory / "model_metadata.json"
    metadata_path = _local_path(baseline["model_metadata_path"], root, "frozen_baseline.model_metadata_path")
    if metadata_path != expected_metadata:
        _fail("frozen_baseline.model_metadata_path must be exactly <run>/model_metadata.json.")

    expected_base_weights = root / "models" / "yolov8n.pt"
    base_weights_path = _local_path(baseline["base_weights_path"], root, "frozen_baseline.base_weights_path")
    if base_weights_path != expected_base_weights:
        _fail("frozen_baseline.base_weights_path must be exactly models/yolov8n.pt.")

    task = baseline["task"]
    if task != "detection":
        _fail("frozen_baseline.task must be exactly 'detection'.")

    class_mapping = _normalise_class_mapping(baseline["class_mapping"], "frozen_baseline.class_mapping")
    if class_mapping != _EXPECTED_CLASS_MAPPING:
        _fail("frozen_baseline.class_mapping must be exactly {'0': 'pothole'}.")

    git_sha = baseline["git_sha"]
    if not isinstance(git_sha, str) or not _GIT_SHA_RE.fullmatch(git_sha):
        _fail("frozen_baseline.git_sha must be a lowercase 40-character Git SHA.")

    dataset_fingerprint = _require_sha256(
        baseline["dataset_fingerprint"], "frozen_baseline.dataset_fingerprint"
    )

    artifact_hashes_raw = baseline["artifact_hashes"]
    if not isinstance(artifact_hashes_raw, dict):
        _fail("frozen_baseline.artifact_hashes must be a mapping.")
    _require_exact_keys(artifact_hashes_raw, set(REQUIRED_ARTIFACT_KEYS), "frozen_baseline.artifact_hashes")
    artifact_hashes = {
        key: _require_sha256(value, f"frozen_baseline.artifact_hashes[{key!r}]")
        for key, value in artifact_hashes_raw.items()
    }

    checkpoint_sha256 = _require_sha256(
        baseline["checkpoint_sha256"], "frozen_baseline.checkpoint_sha256"
    )
    if checkpoint_sha256 != artifact_hashes["weights/best.pt"]:
        _fail("frozen_baseline.checkpoint_sha256 must match artifact_hashes['weights/best.pt'].")

    return FrozenBaselineConfig(
        config_path=path,
        repo_root=root,
        run_id=run_id,
        run_directory=run_directory,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        model_metadata_path=metadata_path,
        model_metadata_sha256=_require_sha256(
            baseline["model_metadata_sha256"], "frozen_baseline.model_metadata_sha256"
        ),
        base_weights_path=base_weights_path,
        task=task,
        class_mapping=MappingProxyType(dict(class_mapping)),
        git_sha=git_sha,
        dataset_fingerprint=dataset_fingerprint,
        artifact_hashes=MappingProxyType(dict(artifact_hashes)),
    )


def verify_frozen_baseline(
    config: FrozenBaselineConfig,
    repo_root: Path | str = REPO_ROOT,
) -> FrozenModelInfo:
    """Fail closed unless the exact frozen local checkpoint is provenance-valid.

    The caller may import Torch/Ultralytics only after this function returns a
    ``FrozenModelInfo``.  The function itself never imports or loads either.
    """

    if not isinstance(config, FrozenBaselineConfig):
        _fail("verify_frozen_baseline requires a FrozenBaselineConfig instance.")

    root = _normalise_repo_root(repo_root)
    if config.repo_root != root:
        _fail("Frozen baseline configuration was loaded for a different repository root.")

    expected_run_directory = root / "outputs" / "training" / config.run_id
    expected_checkpoint = expected_run_directory / "weights" / "best.pt"
    expected_metadata = expected_run_directory / "model_metadata.json"
    expected_base_weights = root / "models" / "yolov8n.pt"
    if (
        config.run_directory != expected_run_directory
        or config.checkpoint_path != expected_checkpoint
        or config.model_metadata_path != expected_metadata
        or config.base_weights_path != expected_base_weights
    ):
        _fail("Frozen baseline configuration paths do not match the required local layout.")

    _ensure_safe_directory(config.run_directory, root, "pinned training run directory")
    checkpoint_sha256 = _sha256(config.checkpoint_path, root, "pinned checkpoint")
    if checkpoint_sha256 != config.checkpoint_sha256:
        _fail("Pinned checkpoint SHA-256 mismatch; refusing to use altered weights.")

    metadata_sha256 = _sha256(config.model_metadata_path, root, "pinned model metadata")
    if metadata_sha256 != config.model_metadata_sha256:
        _fail("Pinned model metadata SHA-256 mismatch; refusing to use altered provenance.")

    metadata = _read_json_mapping(config.model_metadata_path, root, "pinned model metadata")
    if metadata.get("task") != config.task:
        _fail("model_metadata.task does not match the frozen configuration.")
    metadata_class_mapping = _normalise_class_mapping(
        metadata.get("class_mapping"), "model_metadata.class_mapping"
    )
    if metadata_class_mapping != config.class_mapping:
        _fail("model_metadata.class_mapping does not match the frozen configuration.")
    if metadata.get("git_sha") != config.git_sha:
        _fail("model_metadata.git_sha does not match the frozen configuration.")
    if metadata.get("dataset_fingerprint") != config.dataset_fingerprint:
        _fail("model_metadata.dataset_fingerprint does not match the frozen configuration.")
    _strict_metadata_absolute_path(
        metadata.get("run_directory"), config.run_directory, "model_metadata.run_directory"
    )

    metadata_base_weights = metadata.get("base_weights")
    if metadata_base_weights != "models/yolov8n.pt":
        _fail("model_metadata.base_weights does not match the frozen local base-weight path.")
    metadata_resolved_base_weights = _strict_metadata_absolute_path(
        metadata.get("resolved_base_weights"),
        config.base_weights_path,
        "model_metadata.resolved_base_weights",
    )
    if metadata_resolved_base_weights != config.base_weights_path:
        _fail("model_metadata.resolved_base_weights does not match the frozen local base weights.")

    metadata_artifacts = metadata.get("artifacts")
    if not isinstance(metadata_artifacts, dict):
        _fail("model_metadata.artifacts must be a mapping.")
    _require_exact_keys(metadata_artifacts, set(REQUIRED_ARTIFACT_KEYS), "model_metadata.artifacts")

    for key in REQUIRED_ARTIFACT_KEYS:
        recorded_hash = _require_sha256(metadata_artifacts[key], f"model_metadata.artifacts[{key!r}]")
        pinned_hash = config.artifact_hashes[key]
        if recorded_hash != pinned_hash:
            _fail(f"model_metadata artifact hash for {key!r} does not match frozen configuration.")
        actual_hash = _sha256(_expected_artifact_path(key, config), root, f"artifact {key!r}")
        if actual_hash != recorded_hash:
            _fail(f"Artifact SHA-256 mismatch for {key!r}; refusing to use altered provenance.")

    copied_prep_metadata = _read_json_mapping(
        config.run_directory / "dataset_preparation_metadata.json",
        root,
        "copied dataset preparation metadata",
    )
    if copied_prep_metadata.get("dataset_fingerprint") != config.dataset_fingerprint:
        _fail("Copied dataset preparation metadata does not match the frozen dataset fingerprint.")

    return FrozenModelInfo(
        run_id=config.run_id,
        training_run_directory=config.run_directory,
        checkpoint_path=config.checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        model_metadata_path=config.model_metadata_path,
        model_metadata_sha256=metadata_sha256,
        base_weights_path=config.base_weights_path,
        base_weights_sha256=config.artifact_hashes["base_weights"],
        git_sha=config.git_sha,
        dataset_fingerprint=config.dataset_fingerprint,
        task=config.task,
        class_mapping=MappingProxyType(dict(config.class_mapping)),
        artifact_hashes=MappingProxyType(dict(config.artifact_hashes)),
    )
