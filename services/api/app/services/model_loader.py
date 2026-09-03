"""Model loading and fail-closed provenance verification service."""

from __future__ import annotations

import threading
from typing import Any, Optional, Tuple
from services.api.app.core.config import settings
from src.ml.model_provenance import (
    load_frozen_baseline_config,
    verify_frozen_baseline,
    FrozenBaselineConfig,
    FrozenModelInfo,
    FrozenBaselineVerificationError,
)
from src.ml.local_model_runtime import load_verified_yolo_model

_lock = threading.Lock()
_cached_model: Any = None
_cached_model_info: Optional[FrozenModelInfo] = None
_cached_config: Optional[FrozenBaselineConfig] = None


def get_verified_model_info() -> Tuple[FrozenBaselineConfig, FrozenModelInfo]:
    """Verify local frozen baseline configuration without constructing YOLO."""
    global _cached_config, _cached_model_info
    with _lock:
        if _cached_config is None or _cached_model_info is None:
            config = load_frozen_baseline_config(settings.frozen_config_path)
            model_info = verify_frozen_baseline(config)
            _cached_config = config
            _cached_model_info = model_info
        return _cached_config, _cached_model_info


def get_verified_model() -> Tuple[Any, FrozenModelInfo]:
    """Load the verified YOLO baseline once, fail closed if verification fails."""
    global _cached_model, _cached_model_info
    with _lock:
        config, model_info = get_verified_model_info()
        if _cached_model is None:
            _cached_model = load_verified_yolo_model(
                model_info.checkpoint_path,
                model_info.checkpoint_sha256,
            )
        return _cached_model, model_info
