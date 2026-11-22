"""Exercise the CI asset guard with isolated test sessions, not real media."""

from pathlib import Path

import pytest


pytest_plugins = ("pytester",)
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def guarded_pytester(pytester):
    pytester.makeconftest((REPO_ROOT / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini((REPO_ROOT / "pytest.ini").read_text(encoding="utf-8"))
    return pytester


def test_missing_asset_skips_only_marked_test(guarded_pytester):
    missing = guarded_pytester.path / "absent.mp4"
    guarded_pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.requires_local_assets({str(missing)!r})
        def test_needs_media():
            pytest.fail("Missing media must skip before executing the test")

        def test_synthetic_control():
            assert True
        """
    )
    result = guarded_pytester.runpytest_subprocess("-q", "-rs")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*Required local asset files unavailable:*absent.mp4*"])


def test_present_asset_does_not_hide_assertion_failure(guarded_pytester):
    asset = guarded_pytester.path / "present.mp4"
    asset.write_bytes(b"synthetic fixture, no decoder needed")
    guarded_pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.requires_local_assets({str(asset)!r})
        def test_present_asset():
            assert False, "The original assertion must still execute"
        """
    )
    result = guarded_pytester.runpytest_subprocess("-q")
    result.assert_outcomes(failed=1, skipped=0)
    result.stdout.fnmatch_lines(["*The original assertion must still execute*"])


def test_directory_cannot_satisfy_required_asset_file(guarded_pytester):
    asset = guarded_pytester.path / "directory.mp4"
    asset.mkdir()
    guarded_pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.requires_local_assets({str(asset)!r})
        def test_not_a_file():
            pytest.fail("A directory is not an available media file")
        """
    )
    result = guarded_pytester.runpytest_subprocess("-q", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*Required local asset files unavailable:*directory.mp4*"])
