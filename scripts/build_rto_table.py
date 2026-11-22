"""Build the RTO code table from a saved Wikipedia wikitext revision (deterministic).

Input: raw wikitext of "List of Regional Transport Office districts in India" and the
revision JSON returned by the MediaWiki API (both saved locally beforehand; this script
makes no network requests). Output: ``data/reference/in_rto_codes.json`` with one entry
per (state code, RTO number), the office location, jurisdiction, notes and source table
caption, plus the source revision.

Content source: Wikipedia, licensed CC BY-SA 4.0; attribution is stored in the output.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
PAGE_TITLE = "List of Regional Transport Office districts in India"
SECTION = re.compile(r"^==\s*([A-Z]{2})\s*[—–-]\s*(.+?)\s*==\s*$")
CODE = re.compile(r"\b([A-Z]{2})\s?[-–]?\s?(\d{1,2})\b")


def clean(text: str) -> str:
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    for _ in range(3):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s([^\]]*)\]", r"\1", text)
    text = re.sub(r"<br\s*/?>", "; ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("'''", "").replace("''", "").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text).strip(" ;")
    return re.sub(r"^(Location|Jurisdiction|Office)\s*:\s*", "", text, flags=re.I)


def _split_attrs(cell: str):
    """Return (rowspan, content) for a wikitable cell 'attrs | content' or 'content'."""
    rowspan = 1
    if "|" in cell and not cell.strip().startswith("[["):
        head, _, tail = cell.partition("|")
        if "=" in head and "[[" not in head and "{{" not in head:
            match = re.search(r'rowspan\s*=\s*"?(\d+)', head)
            rowspan = int(match.group(1)) if match else 1
            cell = tail
    return rowspan, cell


def parse_tables(lines: List[str]) -> List[dict]:
    """Parse every wikitable in a section into header names and expanded rows."""
    tables, i = [], 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("{|"):
            i += 1
            continue
        headers: List[str] = []
        caption = ""
        raw_rows: List[List[tuple]] = []
        current: Optional[List[tuple]] = None
        i += 1
        while i < len(lines) and not lines[i].lstrip().startswith("|}"):
            line = lines[i].rstrip("\n")
            stripped = line.lstrip()
            if stripped.startswith("|-"):
                if current:
                    raw_rows.append(current)
                current = []
            elif stripped.startswith("!") and "scope=row" in stripped.replace(" ", "").replace('"', ""):
                # Row-header cell (e.g. Gujarat writes "! scope=row | GJ-01"): data, not a column name.
                if current is None:
                    current = []
                current.append(_split_attrs(stripped[1:]))
            elif stripped.startswith("!"):
                for part in re.split(r"!!|\|\|", stripped[1:]):
                    headers.append(clean(part.rpartition("|")[2] if "=" in part.partition("|")[0] else part))
            elif stripped.startswith("|+"):
                caption = clean(stripped[2:])
            elif stripped.startswith("|") and not stripped.startswith("|+"):
                if current is None:
                    current = []
                for part in stripped[1:].split("||"):
                    current.append(_split_attrs(part))
            elif current:
                # continuation of the previous cell's content on a new line
                span, content = current[-1]
                current[-1] = (span, content + " " + line)
            i += 1
        if current:
            raw_rows.append(current)
        # Expand rowspans into a full grid.
        carry: Dict[int, tuple] = {}
        rows = []
        for raw in raw_rows:
            row, col, cells = [], 0, list(raw)
            while cells or col in carry:
                if col in carry:
                    remaining, content = carry[col]
                    row.append(content)
                    if remaining > 1:
                        carry[col] = (remaining - 1, content)
                    else:
                        del carry[col]
                else:
                    span, content = cells.pop(0)
                    row.append(content)
                    if span > 1:
                        carry[col] = (span - 1, content)
                col += 1
            rows.append(row)
        tables.append({"headers": headers, "rows": rows, "caption": caption})
        i += 1
    return tables


def _column(headers: Sequence[str], *needles: str) -> Optional[int]:
    for index, name in enumerate(headers):
        if any(n in name.lower() for n in needles):
            return index
    return None


def build(wikitext: str, revision: dict) -> dict:
    sections: Dict[str, dict] = {}
    current = None
    for line in wikitext.splitlines():
        match = SECTION.match(line)
        if match:
            current = match.group(1)
            sections[current] = {"name": clean(match.group(2)), "lines": []}
        elif line.startswith("==") and current:
            current = None
        elif current:
            sections[current]["lines"].append(line)

    entries: Dict[str, dict] = {}
    report = {}
    for state, section in sections.items():
        codes_here = 0
        for table in parse_tables(section["lines"]):
            headers = table["headers"]
            c_code = _column(headers, "code")
            if c_code is None:
                continue
            c_office = _column(headers, "office") if _column(headers, "office") is not None else _column(headers, "location")
            c_area = _column(headers, "jurisdiction", "area")
            if c_area is None:
                c_area = _column(headers, "district")
            c_note = _column(headers, "annotation", "remark", "note")
            c_type = _column(headers, "type")
            for row in table["rows"]:
                if c_code >= len(row):
                    continue
                raw_code = row[c_code]
                get = lambda c: clean(row[c]) if c is not None and c < len(row) else ""
                for prefix, number in CODE.findall(clean(raw_code)):
                    if prefix != state:
                        continue
                    key = f"{state}-{int(number):02d}"
                    record = entries.setdefault(key, {
                        "code": key, "state_code": state, "state": section["name"], "rto_number": int(number),
                        "office": get(c_office), "office_type": get(c_type), "jurisdiction": [],
                        "notes": get(c_note), "table": table["caption"],
                    })
                    area = get(c_area)
                    if area and area not in record["jurisdiction"]:
                        record["jurisdiction"].append(area)
                    codes_here += 1
        report[state] = {"name": section["name"], "codes": len([k for k in entries if k.startswith(state + "-")])}

    page = next(iter(revision["query"]["pages"].values()))
    rev = page["revisions"][0]
    return {
        "schema_version": "1.0",
        "source": {"title": PAGE_TITLE, "revision_id": rev["revid"], "revision_timestamp": rev["timestamp"],
                   "url": f"https://en.wikipedia.org/w/index.php?title={PAGE_TITLE.replace(' ', '_')}&oldid={rev['revid']}",
                   "license": "CC BY-SA 4.0", "attribution": "Wikipedia contributors"},
        "note": ("Public directory of RTO offices by registration code. It identifies where a series was "
                 "issued, not a vehicle or its owner. Codes absent from the source are not guessed."),
        "coverage": report,
        "entries": dict(sorted(entries.items())),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wikitext", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reference" / "in_rto_codes.json")
    args = parser.parse_args(argv)
    table = build(args.wikitext.read_text(encoding="utf-8"), json.loads(args.revision.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"entries": len(table["entries"]), "coverage": table["coverage"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
