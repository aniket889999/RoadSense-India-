"""Offline lookup of where a registration series was issued (state/UT and RTO office).

The table ``data/reference/in_rto_codes.json`` is built by ``scripts/build_rto_table.py``
from a pinned Wikipedia revision (CC BY-SA 4.0). It is a public office directory: it says
which RTO issues a code, never anything about a vehicle or its owner. No network access.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

from src.anpr.contracts import RegistrationOrigin

TABLE_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "in_rto_codes.json"
_STATE_SERIES = re.compile(r"([A-Z]{2})([0-9]{1,2})[A-Z]{1,3}[0-9]{4}", re.ASCII)
MAX_TEXT = 240

# Codes still seen on older vehicles but no longer issued; names are fixed facts.
LEGACY_STATE_NAMES = {
    "TS": "Telangana (legacy code; current code TG since March 2024)",
    "DN": "Dadra and Nagar Haveli (legacy code; merged UT uses DD since 2020)",
    "OR": "Odisha (legacy code; current code OD)",
    "UA": "Uttarakhand (legacy code; current code UK)",
}
# Present in IN_PLATE_V2 but without a section in the pinned source revision.
MISSING_STATE_NAMES = {"AN": "Andaman and Nicobar Islands"}


@lru_cache(maxsize=1)
def load_table(path: Path = TABLE_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _short(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1].rstrip() + "…"


def lookup_origin(text: str) -> RegistrationOrigin | None:
    """Decode a STATE_SERIES text's state and RTO; ``None`` if not a state series or no table."""
    match = _STATE_SERIES.fullmatch(text)
    table = load_table()
    if match is None or table is None:
        return None
    state_code, number = match.group(1), int(match.group(2))
    rto_code = f"{state_code}-{number:02d}"
    entry = table["entries"].get(rto_code)
    state_name = (entry["state"] if entry else
                  table["coverage"].get(state_code, {}).get("name")
                  or LEGACY_STATE_NAMES.get(state_code) or MISSING_STATE_NAMES.get(state_code))
    return RegistrationOrigin(
        state_code=state_code,
        state_name=state_name,
        rto_code=rto_code,
        rto_in_table=entry is not None,
        rto_office=_short(entry["office"]) if entry and entry["office"] else None,
        rto_jurisdiction=_short("; ".join(entry["jurisdiction"])) if entry and entry["jurisdiction"] else None,
        rto_notes=_short("; ".join(x for x in (entry["notes"], entry["table"]) if x)) if entry else None,
        source_revision=table["source"]["revision_id"],
    )
