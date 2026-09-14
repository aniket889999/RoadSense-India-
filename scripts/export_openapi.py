"""Export the canonical FastAPI OpenAPI document for review and drift checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.app.main import app  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "docs" / "api" / "openapi.json"


def export_openapi(output_path: Path) -> None:
    """Write OpenAPI JSON atomically with deterministic formatting."""
    resolved_output = output_path.resolve()
    docs_root = (REPO_ROOT / "docs").resolve()
    try:
        resolved_output.relative_to(docs_root)
    except ValueError as exc:
        raise ValueError("OpenAPI output must remain inside the repository docs directory.") from exc

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_output.with_name(f".{resolved_output.name}.tmp")
    document = json.dumps(app.openapi(), indent=2) + "\n"
    temporary_path.write_text(document, encoding="utf-8")
    temporary_path.replace(resolved_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_openapi(args.output)


if __name__ == "__main__":
    main()
