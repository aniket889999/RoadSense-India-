"""GPS track import and road-event geolocation for drive sessions.

A dashcam video carries no trustworthy position by itself. This module accepts a GPS
track recorded alongside the drive (GPX, NMEA 0183 or CSV), aligns it to the video
timeline, and places each road event by linear interpolation between the two nearest
track points. Positions are only as good as the track and its time alignment; events
outside the track's time range are left unplaced rather than extrapolated.

Track point format stored on the session (``route_telemetry``)::

    {"timestamp": <seconds from video start>, "lat": <deg>, "lon": <deg>, "utc": <ISO-8601 or None>}
"""

from __future__ import annotations

import csv
import io
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Iterable, Optional, Sequence

MAX_TRACK_BYTES = 5 * 1024 * 1024
MAX_STORED_POINTS = 20_000
# Events may sit this many seconds outside the track's time range and still be placed
# at the nearest end point; beyond that they stay unplaced.
EDGE_TOLERANCE_SECONDS = 2.0
# A gap between consecutive track points longer than this is treated as signal loss:
# events falling inside it are not interpolated across the gap.
MAX_INTERPOLATION_GAP_SECONDS = 30.0

LOCATION_METHOD = "interpolated_from_uploaded_gps_track"


class GpsTrackError(ValueError):
    """Raised when a GPS track cannot be parsed or fails validation."""


@dataclass(frozen=True)
class TrackPoint:
    t: float  # seconds relative to video start, after alignment
    lat: float
    lon: float
    utc: Optional[datetime] = None

    def to_json(self) -> dict:
        return {
            "timestamp": round(self.t, 3),
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "utc": self.utc.isoformat() if self.utc else None,
        }


@dataclass(frozen=True)
class ParsedTrack:
    points: list[TrackPoint]
    source_format: str
    has_absolute_time: bool
    dropped_points: int


# --------------------------------------------------------------------------- helpers


def _valid_coord(lat: float, lon: float) -> bool:
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False
    # (0, 0) is the usual "no fix" placeholder written by cheap loggers.
    return not (abs(lat) < 1e-9 and abs(lon) < 1e-9)


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _decode_text(raw: bytes) -> str:
    if len(raw) > MAX_TRACK_BYTES:
        raise GpsTrackError("GPS track exceeds the 5 MB limit.")
    if not raw.strip():
        raise GpsTrackError("GPS track file is empty.")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 decodes any byte
            raise GpsTrackError("GPS track is not a text file.") from exc


# --------------------------------------------------------------------------- GPX


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_gpx(text: str) -> tuple[list[tuple[Optional[datetime], float, float]], int]:
    # Refuse DTDs/entities outright: a GPS track never needs them and they enable
    # entity-expansion attacks against XML parsers.
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", text, flags=re.IGNORECASE):
        raise GpsTrackError("GPX files with DOCTYPE or ENTITY declarations are not accepted.")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise GpsTrackError(f"GPX is not valid XML: {exc}") from exc
    if _local(root.tag) != "gpx":
        raise GpsTrackError("XML root element is not <gpx>.")
    rows: list[tuple[Optional[datetime], float, float]] = []
    dropped = 0
    for el in root.iter():
        if _local(el.tag) not in {"trkpt", "rtept"}:
            continue
        try:
            lat = float(el.attrib["lat"])
            lon = float(el.attrib["lon"])
        except (KeyError, ValueError):
            dropped += 1
            continue
        when: Optional[datetime] = None
        for child in el:
            if _local(child.tag) == "time" and child.text:
                try:
                    when = _parse_iso(child.text)
                except ValueError:
                    when = None
        if not _valid_coord(lat, lon):
            dropped += 1
            continue
        rows.append((when, lat, lon))
    return rows, dropped


# --------------------------------------------------------------------------- NMEA


def _nmea_checksum_ok(sentence: str) -> bool:
    if "*" not in sentence:
        return True  # some loggers omit checksums; accept but still validate fields
    body, _, given = sentence[1:].partition("*")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(given[:2], 16)
    except ValueError:
        return False


def _nmea_coord(value: str, hemi: str, is_lat: bool) -> float:
    if not value or not hemi:
        raise ValueError("empty coordinate")
    deg_len = 2 if is_lat else 3
    degrees = float(value[:deg_len])
    minutes = float(value[deg_len:])
    if minutes >= 60:
        raise ValueError("minutes out of range")
    out = degrees + minutes / 60.0
    if hemi in {"S", "W"}:
        out = -out
    elif hemi not in {"N", "E"}:
        raise ValueError("bad hemisphere")
    return out


def _nmea_time(hhmmss: str) -> time:
    h, m = int(hhmmss[0:2]), int(hhmmss[2:4])
    s = float(hhmmss[4:])
    whole = int(s)
    return time(h, m, whole, int(round((s - whole) * 1_000_000)) % 1_000_000, tzinfo=timezone.utc)


def parse_nmea(text: str) -> tuple[list[tuple[Optional[datetime], float, float]], int]:
    rows: list[tuple[Optional[datetime], float, float]] = []
    dropped = 0
    current_date: Optional[date] = None
    seen_rmc_times: set[str] = set()
    pending_gga: list[tuple[str, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("$") or len(line) < 7:
            continue
        if not _nmea_checksum_ok(line):
            dropped += 1
            continue
        fields = line.split("*", 1)[0].split(",")
        kind = fields[0][3:]
        try:
            if kind == "RMC" and len(fields) >= 10:
                if fields[2] != "A":  # V = no valid fix
                    dropped += 1
                    continue
                lat = _nmea_coord(fields[3], fields[4], True)
                lon = _nmea_coord(fields[5], fields[6], False)
                d = fields[9]
                current_date = date(2000 + int(d[4:6]), int(d[2:4]), int(d[0:2]))
                when = datetime.combine(current_date, _nmea_time(fields[1]))
                if not _valid_coord(lat, lon):
                    dropped += 1
                    continue
                rows.append((when, lat, lon))
                seen_rmc_times.add(fields[1])
            elif kind == "GGA" and len(fields) >= 7:
                if fields[6] in {"", "0"}:
                    dropped += 1
                    continue
                lat = _nmea_coord(fields[2], fields[3], True)
                lon = _nmea_coord(fields[4], fields[5], False)
                if _valid_coord(lat, lon):
                    pending_gga.append((fields[1], lat, lon))
                else:
                    dropped += 1
        except (ValueError, IndexError):
            dropped += 1
    # GGA carries no date. Use it only when a file has no usable RMC sentences, with
    # the RMC date if one was seen, otherwise as time-of-day relative to the first fix.
    if not rows and pending_gga:
        base = current_date or date(2000, 1, 1)
        for hhmmss, lat, lon in pending_gga:
            try:
                rows.append((datetime.combine(base, _nmea_time(hhmmss)), lat, lon))
            except (ValueError, IndexError):
                dropped += 1
    return rows, dropped


# --------------------------------------------------------------------------- CSV

_LAT_KEYS = ("lat", "latitude")
_LON_KEYS = ("lon", "lng", "long", "longitude")
_REL_KEYS = ("t", "time_s", "seconds", "elapsed_s", "video_seconds", "offset_s")
_ABS_KEYS = ("timestamp", "time", "datetime", "utc", "time_utc")


def parse_csv(text: str) -> tuple[list[tuple[object, float, float]], int, bool]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise GpsTrackError("CSV has no header row.")
    cols = {name.strip().lower(): name for name in reader.fieldnames if name}
    lat_col = next((cols[k] for k in _LAT_KEYS if k in cols), None)
    lon_col = next((cols[k] for k in _LON_KEYS if k in cols), None)
    rel_col = next((cols[k] for k in _REL_KEYS if k in cols), None)
    abs_col = next((cols[k] for k in _ABS_KEYS if k in cols), None)
    if not lat_col or not lon_col:
        raise GpsTrackError("CSV needs latitude and longitude columns (lat/latitude, lon/lng/longitude).")
    if not rel_col and not abs_col:
        raise GpsTrackError(
            "CSV needs a time column: seconds from video start (t, seconds, time_s) or an ISO timestamp (timestamp, time, utc)."
        )
    rows: list[tuple[object, float, float]] = []
    dropped = 0
    for rec in reader:
        try:
            lat = float(rec[lat_col])
            lon = float(rec[lon_col])
            if rel_col:
                when: object = float(rec[rel_col])
                if not math.isfinite(when):  # type: ignore[arg-type]
                    raise ValueError
            else:
                when = _parse_iso(rec[abs_col] or "")
        except (TypeError, ValueError, KeyError):
            dropped += 1
            continue
        if not _valid_coord(lat, lon):
            dropped += 1
            continue
        rows.append((when, lat, lon))
    return rows, dropped, rel_col is None


# --------------------------------------------------------------------------- public API


def detect_format(filename: str, text: str) -> str:
    name = (filename or "").lower()
    head = text.lstrip()[:200].lower()
    if name.endswith(".gpx") or head.startswith("<?xml") or head.startswith("<gpx"):
        return "gpx"
    if name.endswith((".nmea", ".nma", ".log")) or head.startswith("$gp") or head.startswith("$gn"):
        return "nmea"
    if name.endswith((".csv", ".txt")):
        return "nmea" if "$gprmc" in text[:4000].lower() or "$gnrmc" in text[:4000].lower() else "csv"
    raise GpsTrackError("Unsupported GPS track format. Upload a .gpx, .nmea or .csv file.")


def _thin(points: list[TrackPoint], limit: int) -> list[TrackPoint]:
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]


def parse_track(raw: bytes, filename: str, offset_seconds: float = 0.0) -> ParsedTrack:
    """Parse and align a GPS track.

    ``offset_seconds`` is the video time at which the track's first point was recorded
    (positive when the logger started after the video, negative when before). For CSV
    tracks with a relative-seconds column the offset is added to every value.
    """
    if not math.isfinite(offset_seconds) or abs(offset_seconds) > 86_400:
        raise GpsTrackError("offset_seconds must be a finite number within ±24 hours.")
    text = _decode_text(raw)
    fmt = detect_format(filename, text)

    has_abs = True
    if fmt == "gpx":
        rows, dropped = parse_gpx(text)
    elif fmt == "nmea":
        rows, dropped = parse_nmea(text)
    else:
        rows, dropped, has_abs = parse_csv(text)

    if fmt == "gpx" and rows and any(r[0] is None for r in rows):
        raise GpsTrackError("Every GPX track point needs a <time> element to align it with the video.")

    points: list[TrackPoint] = []
    if has_abs:
        abs_rows = sorted((r for r in rows if isinstance(r[0], datetime)), key=lambda r: r[0])
        if abs_rows:
            t0 = abs_rows[0][0]
            for when, lat, lon in abs_rows:
                points.append(TrackPoint((when - t0).total_seconds() + offset_seconds, lat, lon, when))
    else:
        for when, lat, lon in sorted(rows, key=lambda r: r[0]):
            points.append(TrackPoint(float(when) + offset_seconds, lat, lon, None))

    # Drop duplicate timestamps (keep first) and physically impossible jumps (> 120 m/s).
    cleaned: list[TrackPoint] = []
    for p in points:
        if cleaned:
            prev = cleaned[-1]
            dt = p.t - prev.t
            if dt <= 0:
                dropped += 1
                continue
            if _haversine_m(prev.lat, prev.lon, p.lat, p.lon) / dt > 120.0:
                dropped += 1
                continue
        cleaned.append(p)

    if len(cleaned) < 2:
        raise GpsTrackError("GPS track needs at least two valid, time-stamped points.")

    return ParsedTrack(_thin(cleaned, MAX_STORED_POINTS), fmt, has_abs, dropped)


def points_from_json(telemetry: Optional[Sequence[dict]]) -> list[TrackPoint]:
    out: list[TrackPoint] = []
    for item in telemetry or []:
        try:
            out.append(TrackPoint(float(item["timestamp"]), float(item["lat"]), float(item["lon"])))
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda p: p.t)
    return out


def locate(points: Sequence[TrackPoint], t: float) -> Optional[tuple[float, float]]:
    """Interpolate the position at video time ``t``; None if outside the track or in a signal gap."""
    if len(points) < 2 or not math.isfinite(t):
        return None
    first, last = points[0], points[-1]
    if t < first.t:
        return (first.lat, first.lon) if first.t - t <= EDGE_TOLERANCE_SECONDS else None
    if t > last.t:
        return (last.lat, last.lon) if t - last.t <= EDGE_TOLERANCE_SECONDS else None
    lo, hi = 0, len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = points[lo], points[hi]
    span = b.t - a.t
    if span <= 0:
        return (a.lat, a.lon)
    if span > MAX_INTERPOLATION_GAP_SECONDS:
        return None
    f = (t - a.t) / span
    return (a.lat + (b.lat - a.lat) * f, a.lon + (b.lon - a.lon) * f)


def event_time(first_seen: float, last_seen: float) -> float:
    return (float(first_seen) + float(last_seen)) / 2.0


def track_summary(points: Sequence[TrackPoint]) -> dict:
    length = sum(_haversine_m(a.lat, a.lon, b.lat, b.lon) for a, b in zip(points, points[1:]))
    return {
        "point_count": len(points),
        "start_seconds": round(points[0].t, 3) if points else None,
        "end_seconds": round(points[-1].t, 3) if points else None,
        "length_m": round(length, 1),
        "bbox": [
            min(p.lon for p in points), min(p.lat for p in points),
            max(p.lon for p in points), max(p.lat for p in points),
        ] if points else None,
    }


def event_feature(event, lat: float, lon: float, method: str = LOCATION_METHOD) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
        "properties": {
            "kind": "road_event",
            "event_id": event.id,
            "session_id": event.session_id,
            "review_status": event.review_status,
            "model_confidence": round(float(event.representative_confidence), 4),
            "first_seen_seconds": round(float(event.first_seen_seconds), 3),
            "last_seen_seconds": round(float(event.last_seen_seconds), 3),
            "support_count": int(event.support_count),
            "location_method": method,
            "note": "Unverified model suggestion unless review_status is CONFIRMED. No depth or severity is claimed.",
        },
    }


def route_feature(session_id: str, points: Iterable[TrackPoint]) -> dict:
    pts = list(points)
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[round(p.lon, 7), round(p.lat, 7)] for p in pts]},
        "properties": {"kind": "route", "session_id": session_id, **track_summary(pts)},
    }
