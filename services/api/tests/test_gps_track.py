"""GPS track import, alignment and road-event geolocation."""

from __future__ import annotations

import io
import math

import pytest

from services.api.app.services import gps_track as gt

GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="18.5204" lon="73.8567"><time>2026-11-20T06:00:00Z</time></trkpt>
    <trkpt lat="18.5214" lon="73.8567"><time>2026-11-20T06:00:10Z</time></trkpt>
    <trkpt lat="18.5224" lon="73.8567"><time>2026-11-20T06:00:20Z</time></trkpt>
  </trkseg></trk>
</gpx>"""


def _nmea(sentence_body: str) -> str:
    cs = 0
    for ch in sentence_body:
        cs ^= ord(ch)
    return f"${sentence_body}*{cs:02X}"


NMEA = "\n".join(
    [
        _nmea("GPRMC,060000.00,A,1831.224,N,07351.402,E,10.0,0.0,201126,,,A"),
        _nmea("GPRMC,060005.00,A,1831.284,N,07351.402,E,10.0,0.0,201126,,,A"),
        _nmea("GPRMC,060010.00,V,,,,,,,201126,,,N"),  # no fix
        _nmea("GPRMC,060015.00,A,1831.404,N,07351.402,E,10.0,0.0,201126,,,A"),
    ]
)

CSV_REL = "t,lat,lon\n0,18.5204,73.8567\n5,18.5209,73.8567\n10,18.5214,73.8567\n"


def test_gpx_parse_and_align():
    track = gt.parse_track(GPX.encode(), "drive.gpx", offset_seconds=2.0)
    assert track.source_format == "gpx"
    assert [p.t for p in track.points] == [2.0, 12.0, 22.0]
    assert track.points[0].utc is not None


def test_gpx_rejects_doctype():
    evil = GPX.replace('<?xml version="1.0" encoding="UTF-8"?>', '<?xml version="1.0"?><!DOCTYPE gpx [<!ENTITY a "x">]>')
    with pytest.raises(gt.GpsTrackError, match="DOCTYPE"):
        gt.parse_track(evil.encode(), "drive.gpx")


def test_gpx_requires_time():
    no_time = GPX.replace("<time>2026-11-20T06:00:10Z</time>", "")
    with pytest.raises(gt.GpsTrackError, match="time"):
        gt.parse_track(no_time.encode(), "drive.gpx")


def test_nmea_rmc_skips_void_fix_and_converts_minutes():
    track = gt.parse_track(NMEA.encode(), "drive.nmea")
    assert track.source_format == "nmea"
    assert [p.t for p in track.points] == [0.0, 5.0, 15.0]
    assert track.points[0].lat == pytest.approx(18 + 31.224 / 60, abs=1e-9)
    assert track.points[0].lon == pytest.approx(73 + 51.402 / 60, abs=1e-9)
    assert track.dropped_points >= 1


def test_nmea_bad_checksum_dropped():
    bad = NMEA.splitlines()[0][:-2] + "00"
    text = "\n".join([bad] + NMEA.splitlines()[1:])
    track = gt.parse_track(text.encode(), "drive.nmea")
    assert len(track.points) == 2


def test_csv_relative_seconds_with_offset():
    track = gt.parse_track(CSV_REL.encode(), "track.csv", offset_seconds=-1.5)
    assert not track.has_absolute_time
    assert [p.t for p in track.points] == [-1.5, 3.5, 8.5]


def test_csv_missing_columns():
    with pytest.raises(gt.GpsTrackError, match="latitude"):
        gt.parse_track(b"t,x,y\n0,1,2\n1,1,2\n", "track.csv")


def test_rejects_zero_island_and_impossible_jump():
    text = "t,lat,lon\n0,0,0\n1,18.52,73.85\n2,18.5201,73.85\n3,28.6,77.2\n4,18.5203,73.85\n"
    track = gt.parse_track(text.encode(), "track.csv")
    assert len(track.points) == 3  # (0,0) placeholder and the 1,100 km jump removed


def test_too_few_points():
    with pytest.raises(gt.GpsTrackError, match="at least two"):
        gt.parse_track(b"t,lat,lon\n0,18.52,73.85\n", "track.csv")


def test_size_limit():
    with pytest.raises(gt.GpsTrackError, match="5 MB"):
        gt.parse_track(b"a" * (gt.MAX_TRACK_BYTES + 1), "track.csv")


def test_locate_interpolates_and_respects_edges_and_gaps():
    pts = [gt.TrackPoint(0, 18.0, 73.0), gt.TrackPoint(10, 18.001, 73.0), gt.TrackPoint(100, 18.002, 73.0)]
    lat, lon = gt.locate(pts, 5.0)
    assert lat == pytest.approx(18.0005) and lon == pytest.approx(73.0)
    assert gt.locate(pts, -1.0) == (18.0, 73.0)  # within edge tolerance
    assert gt.locate(pts, -5.0) is None  # outside the track
    assert gt.locate(pts, 50.0) is None  # 90 s signal gap: not interpolated
    assert gt.locate(pts, float("nan")) is None


def test_thinning_keeps_endpoints():
    pts = [gt.TrackPoint(i, 18 + i * 1e-6, 73.0) for i in range(50)]
    thin = gt._thin(pts, 10)
    assert len(thin) == 10 and thin[0] == pts[0] and thin[-1] == pts[-1]


def test_track_summary_length():
    pts = [gt.TrackPoint(0, 18.0, 73.0), gt.TrackPoint(10, 18.001, 73.0)]
    s = gt.track_summary(pts)
    assert s["point_count"] == 2
    assert s["length_m"] == pytest.approx(111.2, abs=0.5)


# ------------------------------------------------------------------ API


@pytest.mark.asyncio
async def test_gps_track_api_places_events_and_exports_geojson(client, db_session):
    from services.api.app.models.entities import DriveSession, RoadEvent

    session = DriveSession(
        mode="upload", source_filename="drive.mp4", source_hash="c" * 64,
        processing_state="complete", source_duration_seconds=30.0,
    )
    db_session.add(session)
    await db_session.flush()
    inside = RoadEvent(
        session_id=session.id, first_seen_seconds=4.0, last_seen_seconds=6.0, first_frame_index=20,
        last_frame_index=30, representative_confidence=0.6,
        representative_bbox={"x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2}, support_count=3,
    )
    outside = RoadEvent(
        session_id=session.id, first_seen_seconds=28.0, last_seen_seconds=29.0, first_frame_index=140,
        last_frame_index=145, representative_confidence=0.4,
        representative_bbox={"x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2}, support_count=2,
    )
    db_session.add_all([inside, outside])
    await db_session.commit()

    res = await client.post(
        f"/api/v1/sessions/{session.id}/gps-track",
        files={"file": ("track.csv", io.BytesIO(CSV_REL.encode()), "text/csv")},
        data={"offset_seconds": "0"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["events_total"] == 2 and body["events_placed"] == 1
    assert body["warnings"]

    geo = await client.get(f"/api/v1/sessions/{session.id}/geojson")
    assert geo.status_code == 200
    assert geo.headers["content-type"].startswith("application/geo+json")
    fc = geo.json()
    kinds = [f["properties"]["kind"] for f in fc["features"]]
    assert kinds.count("route") == 1 and kinds.count("road_event") == 1
    point = next(f for f in fc["features"] if f["properties"]["kind"] == "road_event")
    lon, lat = point["geometry"]["coordinates"]
    assert lat == pytest.approx(18.5209, abs=1e-6) and lon == pytest.approx(73.8567, abs=1e-6)
    assert fc["properties"]["unplaced_events"] == 1

    overview = await client.get("/api/v1/geo/road-events")
    assert overview.status_code == 200 and overview.json()["properties"]["count"] == 1
    assert (await client.get("/api/v1/geo/road-events?review_status=BOGUS")).status_code == 422

    removed = await client.delete(f"/api/v1/sessions/{session.id}/gps-track")
    assert removed.status_code == 200
    fc2 = (await client.get(f"/api/v1/sessions/{session.id}/geojson")).json()
    assert fc2["features"] == [] and fc2["properties"]["unplaced_events"] == 2


@pytest.mark.asyncio
async def test_gps_track_api_rejects_bad_file(client, db_session):
    from services.api.app.models.entities import DriveSession

    session = DriveSession(mode="upload", source_filename="d.mp4", source_hash="d" * 64, processing_state="complete")
    db_session.add(session)
    await db_session.commit()
    res = await client.post(
        f"/api/v1/sessions/{session.id}/gps-track",
        files={"file": ("track.kml", io.BytesIO(b"<kml></kml>"), "application/xml")},
    )
    assert res.status_code == 422
    missing = await client.post(
        "/api/v1/sessions/nope/gps-track",
        files={"file": ("track.csv", io.BytesIO(CSV_REL.encode()), "text/csv")},
    )
    assert missing.status_code == 404
