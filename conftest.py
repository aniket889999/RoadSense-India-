"""Explicit local-media requirements without changing existing test assertions.

Only the two recorded real-media integration tests need machine-local files.
The remaining suite uses synthetic fixtures or mocked model/device adapters;
it must continue running on asset-free CPU CI workers.
"""

from pathlib import Path

import pytest


LOCAL_ASSET_TESTS = {
    "tests/test_stability_engine.py::test_real_parking_video_read_only_negative_check": (
        "/Users/aniket/Downloads/PARKING LOT TEST.mp4",
        "/Users/aniket/Downloads/parking_reference.jpg",
    ),
    "services/api/tests/test_parking_evidence_workflow.py::test_unstable_negative_video_remains_strictly_blocked": (
        "/Users/aniket/Downloads/PARKING LOT TEST.mp4",
    ),
}


def pytest_collection_modifyitems(items):
    for item in items:
        # Parametrized variants have the same asset requirements.
        nodeid = item.nodeid.split("[", 1)[0]
        paths = LOCAL_ASSET_TESTS.get(nodeid)
        if paths:
            item.add_marker(pytest.mark.requires_local_assets(*paths))


def pytest_runtest_setup(item):
    for marker in item.iter_markers("requires_local_assets"):
        if not marker.args:
            raise pytest.UsageError("requires_local_assets needs at least one file path")
        missing = [str(path) for path in marker.args if not Path(path).is_file()]
        if missing:
            pytest.skip("Required local asset files unavailable: " + ", ".join(missing))
