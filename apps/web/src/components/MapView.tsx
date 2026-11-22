'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { GeoJSONSource, Map as MlMap, MapGeoJSONFeature } from 'maplibre-gl';
import { Download, Info, Loader2, MapPin, Route, Trash2, Upload } from 'lucide-react';
import { DriveSession, RoadEvent } from '../lib/types';
import {
  deleteGpsTrack,
  fetchLocatedRoadEvents,
  fetchSessionGeoJson,
  getSessionGeoJsonUrl,
  GeoFeatureCollection,
  GpsTrackUploadResult,
  uploadGpsTrack,
} from '../lib/api';

interface MapViewProps {
  session: DriveSession | null;
  roadEvents: RoadEvent[];
  onTrackChanged?: () => void;
}

const STATUS_COLORS: Record<string, string> = {
  PENDING_REVIEW: '#D97706',
  CONFIRMED: '#16A34A',
  NEEDS_REVISIT: '#DC2626',
  REJECTED: '#94A3B8',
  SPLITED: '#7C3AED',
  MERGED: '#7C3AED',
};

const STATUS_LABELS: Record<string, string> = {
  PENDING_REVIEW: 'Pending review',
  CONFIRMED: 'Confirmed',
  NEEDS_REVISIT: 'Needs revisit',
  REJECTED: 'Rejected',
};

const INDIA_CENTER: [number, number] = [78.9629, 22.5937];

const OSM_STYLE = {
  version: 8 as const,
  sources: {
    osm: {
      type: 'raster' as const,
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      maxzoom: 19,
      attribution: '© OpenStreetMap contributors',
    },
  },
  layers: [{ id: 'osm', type: 'raster' as const, source: 'osm' }],
};

const EMPTY_FC: GeoFeatureCollection = { type: 'FeatureCollection', features: [] };

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function boundsOf(fc: GeoFeatureCollection): [[number, number], [number, number]] | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const visit = (c: number[]) => {
    minX = Math.min(minX, c[0]); maxX = Math.max(maxX, c[0]);
    minY = Math.min(minY, c[1]); maxY = Math.max(maxY, c[1]);
  };
  for (const f of fc.features) {
    const g = f.geometry;
    if (g.type === 'Point') visit(g.coordinates as number[]);
    if (g.type === 'LineString') (g.coordinates as number[][]).forEach(visit);
  }
  return Number.isFinite(minX) ? [[minX, minY], [maxX, maxY]] : null;
}

export function MapView({ session, roadEvents, onTrackChanged }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);

  const [scope, setScope] = useState<'session' | 'all'>('session');
  const [data, setData] = useState<GeoFeatureCollection>(EMPTY_FC);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set(['REJECTED']));

  const [trackFile, setTrackFile] = useState<File | null>(null);
  const [offset, setOffset] = useState('0');
  const [uploading, setUploading] = useState(false);
  const [lastUpload, setLastUpload] = useState<GpsTrackUploadResult | null>(null);

  // ---------------------------------------------------------------- map setup
  useEffect(() => {
    let cancelled = false;
    let map: MlMap | null = null;
    (async () => {
      try {
        const mod = await import('maplibre-gl');
        // maplibre ships a UMD bundle: depending on interop the API is on the module or its default export.
        const maplibregl = ((mod as unknown as { default?: typeof mod }).default ?? mod) as typeof mod;
        if (cancelled || !containerRef.current) return;
        map = new maplibregl.Map({
          container: containerRef.current,
          style: OSM_STYLE,
          center: INDIA_CENTER,
          zoom: 4,
          attributionControl: { compact: false },
        });
        map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
        map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');
        map.on('load', () => {
          if (!map) return;
          map.addSource('roadsense', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
          map.addLayer({
            id: 'route-casing', type: 'line', source: 'roadsense',
            filter: ['==', ['get', 'kind'], 'route'],
            paint: { 'line-color': '#FFFFFF', 'line-width': 7 },
            layout: { 'line-cap': 'round', 'line-join': 'round' },
          });
          map.addLayer({
            id: 'route', type: 'line', source: 'roadsense',
            filter: ['==', ['get', 'kind'], 'route'],
            paint: { 'line-color': '#4F46E5', 'line-width': 4 },
            layout: { 'line-cap': 'round', 'line-join': 'round' },
          });
          map.addLayer({
            id: 'events', type: 'circle', source: 'roadsense',
            filter: ['==', ['get', 'kind'], 'road_event'],
            paint: {
              'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 5, 16, 9],
              'circle-color': [
                'match', ['get', 'review_status'],
                'CONFIRMED', STATUS_COLORS.CONFIRMED,
                'NEEDS_REVISIT', STATUS_COLORS.NEEDS_REVISIT,
                'REJECTED', STATUS_COLORS.REJECTED,
                'PENDING_REVIEW', STATUS_COLORS.PENDING_REVIEW,
                '#7C3AED',
              ],
              'circle-stroke-color': '#FFFFFF',
              'circle-stroke-width': 2,
            },
          });
          map.on('click', 'events', (e) => {
            const f = e.features?.[0] as MapGeoJSONFeature | undefined;
            if (!f || f.geometry.type !== 'Point' || !map) return;
            const p = f.properties as Record<string, unknown>;
            const status = String(p.review_status);
            new maplibregl.Popup({ offset: 10, maxWidth: '280px' })
              .setLngLat(f.geometry.coordinates as [number, number])
              .setHTML(
                `<div style="font:12px/1.45 system-ui;color:#0F172A">
                   <div style="font-weight:600;margin-bottom:4px">Road event · ${escapeHtml(STATUS_LABELS[status] ?? status)}</div>
                   <div>Video time: ${escapeHtml(Number(p.first_seen_seconds).toFixed(1))}–${escapeHtml(Number(p.last_seen_seconds).toFixed(1))} s</div>
                   <div>Model confidence: ${escapeHtml(Number(p.model_confidence).toFixed(2))} · ${escapeHtml(p.support_count)} frames</div>
                   <div style="color:#64748B;margin-top:4px">Position interpolated from the uploaded GPS track. ${
                     status === 'CONFIRMED' ? 'Confirmed by a reviewer.' : 'Unverified model suggestion.'
                   } No depth or severity is claimed.</div>
                 </div>`
              )
              .addTo(map);
          });
          map.on('mouseenter', 'events', () => { if (map) map.getCanvas().style.cursor = 'pointer'; });
          map.on('mouseleave', 'events', () => { if (map) map.getCanvas().style.cursor = ''; });
          setMapReady(true);
        });
        map.on('error', (e) => {
          const msg = (e?.error as Error | undefined)?.message || '';
          if (/tile|fetch|Failed to load/i.test(msg)) setMapError('Map tiles could not load (offline?). Routes and events are still drawn.');
        });
        mapRef.current = map;
      } catch (err) {
        setMapError(err instanceof Error ? err.message : 'Map failed to initialise.');
      }
    })();
    return () => {
      cancelled = true;
      map?.remove();
      mapRef.current = null;
      setMapReady(false);
    };
  }, []);

  // ---------------------------------------------------------------- data
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (scope === 'all') setData(await fetchLocatedRoadEvents());
      else if (session) setData(await fetchSessionGeoJson(session.id));
      else setData(EMPTY_FC);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load map data.');
      setData(EMPTY_FC);
    } finally {
      setLoading(false);
    }
  }, [scope, session]);

  useEffect(() => { load(); }, [load, roadEvents]);

  const visible = useMemo<GeoFeatureCollection>(() => ({
    type: 'FeatureCollection',
    features: data.features.filter(
      (f) => f.properties?.kind !== 'road_event' || !hidden.has(String(f.properties?.review_status))
    ),
  }), [data, hidden]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const src = map.getSource('roadsense') as GeoJSONSource | undefined;
    src?.setData(visible as unknown as Parameters<GeoJSONSource['setData']>[0]);
  }, [visible, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const b = boundsOf(data);
    if (b) map.fitBounds(b, { padding: 60, maxZoom: 17, duration: 600 });
  }, [data, mapReady]);

  // ---------------------------------------------------------------- stats
  const route = data.features.find((f) => f.properties?.kind === 'route');
  const placedEvents = data.features.filter((f) => f.properties?.kind === 'road_event');
  const counts = placedEvents.reduce<Record<string, number>>((acc, f) => {
    const s = String(f.properties?.review_status);
    acc[s] = (acc[s] ?? 0) + 1;
    return acc;
  }, {});
  const unplaced = Number(data.properties?.unplaced_events ?? 0);
  const hasTrack = Boolean(route);

  const onUpload = async () => {
    if (!session || !trackFile) return;
    const offsetNum = Number(offset);
    if (!Number.isFinite(offsetNum)) { setError('Offset must be a number of seconds.'); return; }
    setUploading(true);
    setError(null);
    try {
      const res = await uploadGpsTrack(session.id, trackFile, offsetNum);
      setLastUpload(res);
      setTrackFile(null);
      await load();
      onTrackChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'GPS track upload failed.');
    } finally {
      setUploading(false);
    }
  };

  const onRemove = async () => {
    if (!session) return;
    setUploading(true);
    try {
      await deleteGpsTrack(session.id);
      setLastUpload(null);
      await load();
      onTrackChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove GPS track.');
    } finally {
      setUploading(false);
    }
  };

  const toggle = (status: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(status)) next.delete(status); else next.add(status);
      return next;
    });

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 h-full">
      {/* Map */}
      <div className="xl:col-span-8 relative rounded-3xl overflow-hidden border border-command-border bg-command-elevated min-h-[420px]">
        <div ref={containerRef} className="absolute inset-0" aria-label="Road events map" role="region" />
        <div className="absolute top-3 left-3 flex gap-1 p-1 rounded-full bg-white/95 border border-command-border shadow-subtle">
          {(['session', 'all'] as const).map((s) => (
            <button
              key={s}
              onClick={() => setScope(s)}
              className={`px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-colors ${
                scope === s ? 'bg-brand text-white' : 'text-command-muted hover:text-command-text'
              }`}
            >
              {s === 'session' ? 'This session' : 'All sessions'}
            </button>
          ))}
        </div>
        {loading && (
          <div className="absolute top-3 right-14 flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/95 border border-command-border text-[12px] text-command-muted">
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading
          </div>
        )}
        {mapError && (
          <div className="absolute bottom-10 left-3 right-3 px-3 py-2 rounded-xl bg-amber-50 border border-amber-200 text-[12px] text-amber-700">
            {mapError}
          </div>
        )}
        {!loading && placedEvents.length === 0 && !hasTrack && (
          <div className="absolute inset-x-6 bottom-12 md:inset-x-auto md:left-1/2 md:-translate-x-1/2 max-w-md px-5 py-4 rounded-2xl bg-white/95 border border-command-border shadow-subtle text-center">
            <p className="text-[14px] font-medium text-command-text">
              {scope === 'all' ? 'No road events have a position yet' : session ? 'No GPS track for this session' : 'No drive session selected'}
            </p>
            <p className="mt-1 text-[12px] text-command-muted">
              {session
                ? 'Upload the GPS log recorded during this drive (GPX, NMEA or CSV). Each road event is placed by its video time.'
                : 'Upload and process a dashcam video under Road inspection, then add its GPS track here.'}
            </p>
          </div>
        )}
      </div>

      {/* Side panel */}
      <aside className="xl:col-span-4 space-y-4 overflow-y-auto">
        <section className="glass-panel p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Route className="w-4 h-4 text-brand" />
            <h2 className="text-[15px] font-medium text-command-text">GPS track</h2>
          </div>
          <p className="text-[12px] text-command-muted truncate">
            {session ? <>Session <span className="font-mono text-command-text">{session.source_filename}</span></> : 'Select a session under Drive sessions.'}
          </p>

          {hasTrack && route ? (
            <dl className="grid grid-cols-2 gap-2 text-[12px]">
              <div className="rounded-xl bg-command-elevated px-3 py-2"><dt className="text-command-muted">Points</dt><dd className="text-command-text font-medium">{Number(route.properties?.point_count).toLocaleString()}</dd></div>
              <div className="rounded-xl bg-command-elevated px-3 py-2"><dt className="text-command-muted">Length</dt><dd className="text-command-text font-medium">{(Number(route.properties?.length_m) / 1000).toFixed(2)} km</dd></div>
              <div className="rounded-xl bg-command-elevated px-3 py-2"><dt className="text-command-muted">Video time</dt><dd className="text-command-text font-medium">{Number(route.properties?.start_seconds).toFixed(0)}–{Number(route.properties?.end_seconds).toFixed(0)} s</dd></div>
              <div className="rounded-xl bg-command-elevated px-3 py-2"><dt className="text-command-muted">Events placed</dt><dd className="text-command-text font-medium">{placedEvents.length}{unplaced ? ` (${unplaced} not)` : ''}</dd></div>
            </dl>
          ) : null}

          <label className="block">
            <span className="text-[12px] text-command-muted">GPS log (.gpx, .nmea, .csv · max 5 MB)</span>
            <input
              type="file"
              accept=".gpx,.nmea,.nma,.log,.csv,.txt"
              disabled={!session || uploading}
              onChange={(e) => setTrackFile(e.target.files?.[0] ?? null)}
              className="mt-1 block w-full text-[12px] text-command-muted file:mr-3 file:px-3 file:py-1.5 file:rounded-full file:border-0 file:bg-brand-soft file:text-brand-deep file:font-medium disabled:opacity-50"
            />
          </label>
          <label className="block">
            <span className="text-[12px] text-command-muted">Video time of the first GPS point (seconds, can be negative)</span>
            <input
              type="number"
              step="0.1"
              value={offset}
              onChange={(e) => setOffset(e.target.value)}
              disabled={!session || uploading}
              className="mt-1 w-full rounded-xl border border-command-border bg-white px-3 py-2 text-[13px] text-command-text focus:outline-none focus:ring-2 focus:ring-brand/30"
            />
          </label>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={onUpload}
              disabled={!session || !trackFile || uploading}
              className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-brand hover:bg-brand-bright text-white text-[13px] font-medium disabled:opacity-40"
            >
              {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              {hasTrack ? 'Replace track' : 'Add track'}
            </button>
            {hasTrack && (
              <button
                onClick={onRemove}
                disabled={uploading}
                className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-command-elevated hover:bg-command-border text-command-text text-[13px] disabled:opacity-40"
              >
                <Trash2 className="w-4 h-4" /> Remove
              </button>
            )}
            {session && (hasTrack || placedEvents.length > 0) && (
              <a
                href={getSessionGeoJsonUrl(session.id)}
                download
                className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-command-text hover:bg-slate-700 text-white text-[13px]"
              >
                <Download className="w-4 h-4" /> GeoJSON
              </a>
            )}
          </div>
          {lastUpload && (
            <div className="rounded-xl bg-command-elevated px-3 py-2 text-[12px] text-command-muted space-y-1">
              <div>
                Read {lastUpload.point_count.toLocaleString()} points ({lastUpload.source_format.toUpperCase()}
                {lastUpload.dropped_points ? `, ${lastUpload.dropped_points} invalid dropped` : ''}); placed {lastUpload.events_placed} of {lastUpload.events_total} events.
              </div>
              {lastUpload.warnings.map((w) => <div key={w} className="text-amber-700">{w}</div>)}
            </div>
          )}
          {error && <div className="rounded-xl bg-rose-50 border border-rose-200 px-3 py-2 text-[12px] text-rose-700">{error}</div>}
        </section>

        <section className="glass-panel p-5 space-y-3">
          <div className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-brand" />
            <h2 className="text-[15px] font-medium text-command-text">Road events on the map</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.keys(STATUS_LABELS).map((s) => (
              <button
                key={s}
                onClick={() => toggle(s)}
                aria-pressed={!hidden.has(s)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-[12px] transition-colors ${
                  hidden.has(s) ? 'border-command-border text-command-muted bg-white' : 'border-command-border bg-command-elevated text-command-text'
                }`}
              >
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: STATUS_COLORS[s], opacity: hidden.has(s) ? 0.35 : 1 }} />
                {STATUS_LABELS[s]} <span className="text-command-muted">{counts[s] ?? 0}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="rounded-3xl border border-command-border bg-white p-5 text-[12px] text-command-muted space-y-2">
          <div className="flex items-center gap-2 text-command-text font-medium"><Info className="w-4 h-4 text-brand" /> How positions are made</div>
          <p>
            Positions come from the GPS log you upload, linearly interpolated at each event&apos;s video time. Events
            outside the track or inside a GPS gap longer than 30 s stay unplaced. Accuracy is limited by the logger and
            by how well the offset aligns it with the video.
          </p>
          <p>Base map © OpenStreetMap contributors. Only map tiles are downloaded; video, events and tracks stay on this machine.</p>
        </section>
      </aside>
    </div>
  );
}
