'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  ApproachZone,
  CalibrationStatus,
  LayoutRevisionStatus,
  LayoutValidationError,
  NormalizedPoint,
  ParkingSpace,
  SpaceType,
} from '../lib/types';
import {
  ViewportBox,
  canvasToImageNormalized,
  computeContainViewport,
  imageNormalizedToCanvas,
  setupHiDPICanvas,
} from '../lib/canvasTransform';

export type DrawingMode = 'idle' | 'draw_space' | 'draw_approach' | 'edit_vertices';

interface ParkingCanvasProps {
  referenceImageUrl: string | null;
  status: LayoutRevisionStatus;
  calibrationStatus: CalibrationStatus;
  parkingSpaces: ParkingSpace[];
  approachZones: ApproachZone[];
  selectedSpaceId: string | null;
  drawingMode: DrawingMode;
  currentPoints: NormalizedPoint[];
  validationErrors?: LayoutValidationError[];
  onSelectSpace: (spaceId: string | null) => void;
  onAddPoint: (point: NormalizedPoint) => void;
  onClosePolygon: () => void;
  onUpdateSpacePolygon: (spaceId: string, points: NormalizedPoint[]) => void;
  onUpdateApproachPolygon: (spaceId: string, points: NormalizedPoint[]) => void;
}

const STATUS_COLORS: Record<LayoutRevisionStatus, { stroke: string; fill: string; badge: string }> = {
  DRAFT: { stroke: '#3B82F6', fill: 'rgba(59, 130, 246, 0.20)', badge: 'bg-blue-500/20 text-blue-400 border-blue-500/30' },
  PENDING_REVIEW: { stroke: '#F59E0B', fill: 'rgba(245, 158, 11, 0.20)', badge: 'bg-amber-500/20 text-amber-400 border-amber-500/30' },
  VERIFIED: { stroke: '#10B981', fill: 'rgba(16, 185, 129, 0.20)', badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' },
  SUPERSEDED: { stroke: '#6B7280', fill: 'rgba(107, 114, 128, 0.15)', badge: 'bg-gray-500/20 text-gray-400 border-gray-500/30' },
  INVALIDATED: { stroke: '#EF4444', fill: 'rgba(239, 68, 68, 0.20)', badge: 'bg-red-500/20 text-red-400 border-red-500/30' },
};

const SELECTED_COLOR = { stroke: '#06B6D4', fill: 'rgba(6, 182, 212, 0.35)' };
const APPROACH_COLOR = { stroke: '#8B5CF6', fill: 'rgba(139, 92, 246, 0.20)' };
const ERROR_COLOR = { stroke: '#EF4444', fill: 'rgba(239, 68, 68, 0.35)' };

export function ParkingCanvas({
  referenceImageUrl,
  status,
  calibrationStatus,
  parkingSpaces,
  approachZones,
  selectedSpaceId,
  drawingMode,
  currentPoints,
  validationErrors = [],
  onSelectSpace,
  onAddPoint,
  onClosePolygon,
  onUpdateSpacePolygon,
  onUpdateApproachPolygon,
}: ParkingCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const [imageLoaded, setImageLoaded] = useState(false);
  const [viewport, setViewport] = useState<ViewportBox>({ x0: 0, y0: 0, width: 0, height: 0 });
  const [activeHandle, setActiveHandle] = useState<{ spaceId: string; isApproach: boolean; vertexIndex: number } | null>(null);

  // Load image
  useEffect(() => {
    if (!referenceImageUrl) {
      imageRef.current = null;
      setImageLoaded(false);
      return;
    }

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = referenceImageUrl;
    img.onload = () => {
      imageRef.current = img;
      setImageLoaded(true);
    };
    img.onerror = () => {
      console.error('Failed to load reference image:', referenceImageUrl);
      setImageLoaded(false);
    };
  }, [referenceImageUrl]);

  // Compute viewport on resize
  const updateDimensions = useCallback(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const cssW = container.clientWidth;
    const cssH = container.clientHeight;
    if (cssW <= 0 || cssH <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    setupHiDPICanvas(canvas, cssW, cssH, dpr);

    const imgW = imageRef.current && imageLoaded ? imageRef.current.naturalWidth : cssW;
    const imgH = imageRef.current && imageLoaded ? imageRef.current.naturalHeight : cssH;

    const vp = computeContainViewport(cssW, cssH, imgW, imgH);
    setViewport(vp);
  }, [imageLoaded]);

  useEffect(() => {
    updateDimensions();
  }, [updateDimensions]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      updateDimensions();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [updateDimensions]);

  // Main Render loop
  const drawScene = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const container = containerRef.current;
    const cssW = container ? container.clientWidth : canvas.width;
    const cssH = container ? container.clientHeight : canvas.height;

    // Clear outer canvas with dark margin color
    ctx.fillStyle = '#0B0F12';
    ctx.fillRect(0, 0, cssW, cssH);

    // 1. Draw Reference Image or Placeholder within letterboxed viewport
    if (imageRef.current && imageLoaded && viewport.width > 0 && viewport.height > 0) {
      ctx.drawImage(
        imageRef.current,
        viewport.x0,
        viewport.y0,
        viewport.width,
        viewport.height
      );
      // Subtle frame outline
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.strokeRect(viewport.x0, viewport.y0, viewport.width, viewport.height);
    } else {
      // Placeholder Grid
      ctx.fillStyle = '#12181F';
      ctx.fillRect(viewport.x0, viewport.y0, viewport.width || cssW, viewport.height || cssH);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < cssW; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, cssH);
        ctx.stroke();
      }
      for (let y = 0; y < cssH; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(cssW, y);
        ctx.stroke();
      }
    }

    if (viewport.width <= 0 || viewport.height <= 0) return;

    const statusStyle = STATUS_COLORS[status] || STATUS_COLORS.DRAFT;
    const errorSpaceIds = new Set(
      validationErrors.map((e) => e.space_id).filter(Boolean) as string[]
    );

    // 2. Draw Approach Zones (Dashed lines)
    approachZones.forEach((az) => {
      if (az.polygon_normalized.length < 3) return;
      const isLinkedToSelected = az.parking_space_id === selectedSpaceId;

      ctx.save();
      ctx.beginPath();
      const p0 = imageNormalizedToCanvas(az.polygon_normalized[0], viewport);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < az.polygon_normalized.length; i++) {
        const p = imageNormalizedToCanvas(az.polygon_normalized[i], viewport);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();

      ctx.setLineDash([6, 4]);
      ctx.lineWidth = isLinkedToSelected ? 2.5 : 1.5;
      ctx.strokeStyle = isLinkedToSelected ? SELECTED_COLOR.stroke : APPROACH_COLOR.stroke;
      ctx.fillStyle = isLinkedToSelected ? SELECTED_COLOR.fill : APPROACH_COLOR.fill;
      ctx.fill();
      ctx.stroke();

      // Vertex Handles for Approach Zone if linked to selected space in DRAFT
      if (isLinkedToSelected && status === 'DRAFT') {
        az.polygon_normalized.forEach((p) => {
          const cp = imageNormalizedToCanvas(p, viewport);
          ctx.beginPath();
          ctx.arc(cp.x, cp.y, 6, 0, Math.PI * 2);
          ctx.fillStyle = '#8B5CF6';
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 2;
          ctx.fill();
          ctx.stroke();
        });
      }
      ctx.restore();
    });

    // 3. Draw Configured Parking Spaces
    parkingSpaces.forEach((sp) => {
      if (sp.polygon_normalized.length < 3) return;
      const isSelected = sp.id === selectedSpaceId;
      const hasError = sp.id ? errorSpaceIds.has(sp.id) : false;

      ctx.save();
      ctx.beginPath();
      const p0 = imageNormalizedToCanvas(sp.polygon_normalized[0], viewport);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < sp.polygon_normalized.length; i++) {
        const p = imageNormalizedToCanvas(sp.polygon_normalized[i], viewport);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();

      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = hasError
        ? ERROR_COLOR.stroke
        : isSelected
        ? SELECTED_COLOR.stroke
        : statusStyle.stroke;
      ctx.fillStyle = hasError
        ? ERROR_COLOR.fill
        : isSelected
        ? SELECTED_COLOR.fill
        : statusStyle.fill;
      ctx.fill();
      ctx.stroke();

      // Centroid for Label
      let cx = 0;
      let cy = 0;
      sp.polygon_normalized.forEach((p) => {
        const cp = imageNormalizedToCanvas(p, viewport);
        cx += cp.x;
        cy += cp.y;
      });
      cx /= sp.polygon_normalized.length;
      cy /= sp.polygon_normalized.length;

      // Draw Badge / Label
      ctx.font = 'bold 11px monospace';
      const labelText = sp.operator_label || 'Bay';
      const textMetrics = ctx.measureText(labelText);
      const padding = 6;
      const boxW = textMetrics.width + padding * 2;
      const boxH = 18;

      ctx.fillStyle = hasError ? '#EF4444' : isSelected ? '#06B6D4' : '#0B0F12';
      ctx.strokeStyle = hasError ? '#FFFFFF' : isSelected ? '#FFFFFF' : statusStyle.stroke;
      ctx.lineWidth = 1;
      ctx.fillRect(cx - boxW / 2, cy - boxH / 2, boxW, boxH);
      ctx.strokeRect(cx - boxW / 2, cy - boxH / 2, boxW, boxH);

      ctx.fillStyle = hasError || isSelected ? '#0B0F12' : '#E2E8F0';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, cx, cy);

      // Vertex Handles if Selected and in DRAFT mode
      if (isSelected && status === 'DRAFT') {
        sp.polygon_normalized.forEach((p) => {
          const cp = imageNormalizedToCanvas(p, viewport);
          ctx.beginPath();
          ctx.arc(cp.x, cp.y, 6, 0, Math.PI * 2);
          ctx.fillStyle = '#06B6D4';
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 2;
          ctx.fill();
          ctx.stroke();
        });
      }
      ctx.restore();
    });

    // 4. Draw Active Drawing Polygon
    if (currentPoints.length > 0) {
      ctx.save();
      ctx.beginPath();
      const p0 = imageNormalizedToCanvas(currentPoints[0], viewport);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < currentPoints.length; i++) {
        const p = imageNormalizedToCanvas(currentPoints[i], viewport);
        ctx.lineTo(p.x, p.y);
      }

      ctx.lineWidth = 2;
      ctx.strokeStyle = drawingMode === 'draw_approach' ? '#A855F7' : '#38BDF8';
      ctx.setLineDash([4, 4]);
      ctx.stroke();

      currentPoints.forEach((p, idx) => {
        const cp = imageNormalizedToCanvas(p, viewport);
        ctx.beginPath();
        ctx.arc(cp.x, cp.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = idx === 0 ? '#10B981' : '#38BDF8';
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
      });
      ctx.restore();
    }
  }, [
    imageLoaded,
    viewport,
    status,
    parkingSpaces,
    approachZones,
    selectedSpaceId,
    drawingMode,
    currentPoints,
    validationErrors,
  ]);

  useEffect(() => {
    drawScene();
  }, [drawScene]);

  // Ray-casting point inside polygon
  const isPointInPolygon = (pt: { x: number; y: number }, poly: { x: number; y: number }[]) => {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i].x, yi = poly[i].y;
      const xj = poly[j].x, yj = poly[j].y;
      const intersect = yi > pt.y !== yj > pt.y && pt.x < ((xj - xi) * (pt.y - yi)) / (yj - yi) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  };

  // Canvas Mouse Interactions
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || viewport.width <= 0 || viewport.height <= 0) return;

    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    // Reject events in letterbox margins
    const normPt = canvasToImageNormalized({ x: canvasX, y: canvasY }, viewport);
    if (!normPt) return;

    if (drawingMode === 'draw_space' || drawingMode === 'draw_approach') {
      if (currentPoints.length >= 3) {
        const startCanvas = imageNormalizedToCanvas(currentPoints[0], viewport);
        const dist = Math.hypot(canvasX - startCanvas.x, canvasY - startCanvas.y);
        if (dist < 15) {
          onClosePolygon();
          return;
        }
      }
      onAddPoint(normPt);
      return;
    }

    // Edit Mode: Check handles for space vertices and approach vertices
    if (selectedSpaceId && status === 'DRAFT') {
      const space = parkingSpaces.find((s) => s.id === selectedSpaceId);
      if (space) {
        for (let i = 0; i < space.polygon_normalized.length; i++) {
          const cp = imageNormalizedToCanvas(space.polygon_normalized[i], viewport);
          if (Math.hypot(canvasX - cp.x, canvasY - cp.y) <= 10) {
            setActiveHandle({ spaceId: space.id!, isApproach: false, vertexIndex: i });
            return;
          }
        }
      }

      const az = approachZones.find((a) => a.parking_space_id === selectedSpaceId);
      if (az) {
        for (let i = 0; i < az.polygon_normalized.length; i++) {
          const cp = imageNormalizedToCanvas(az.polygon_normalized[i], viewport);
          if (Math.hypot(canvasX - cp.x, canvasY - cp.y) <= 10) {
            setActiveHandle({ spaceId: selectedSpaceId, isApproach: true, vertexIndex: i });
            return;
          }
        }
      }
    }

    // Selection Mode
    for (const sp of parkingSpaces) {
      const canvasPoly = sp.polygon_normalized.map((p) => imageNormalizedToCanvas(p, viewport));
      if (isPointInPolygon({ x: canvasX, y: canvasY }, canvasPoly)) {
        onSelectSpace(sp.id || null);
        return;
      }
    }

    onSelectSpace(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!activeHandle) return;
    const canvas = canvasRef.current;
    if (!canvas || viewport.width <= 0 || viewport.height <= 0) return;

    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    const normPt = canvasToImageNormalized({ x: canvasX, y: canvasY }, viewport);
    if (!normPt) return;

    if (!activeHandle.isApproach) {
      const space = parkingSpaces.find((s) => s.id === activeHandle.spaceId);
      if (space) {
        const newPts = [...space.polygon_normalized];
        newPts[activeHandle.vertexIndex] = normPt;
        onUpdateSpacePolygon(space.id!, newPts);
      }
    } else {
      const az = approachZones.find((a) => a.parking_space_id === activeHandle.spaceId);
      if (az) {
        const newPts = [...az.polygon_normalized];
        newPts[activeHandle.vertexIndex] = normPt;
        onUpdateApproachPolygon(activeHandle.spaceId, newPts);
      }
    }
  };

  const handleMouseUp = () => {
    setActiveHandle(null);
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full min-h-[480px] bg-[#0B0F12] rounded-xl border border-command-border overflow-hidden select-none"
    >
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className={`w-full h-full block ${
          drawingMode !== 'idle' ? 'cursor-crosshair' : activeHandle ? 'cursor-grabbing' : 'cursor-default'
        }`}
      />

      {/* Mode Status Overlay Badge */}
      <div className="absolute top-3 left-3 pointer-events-none flex items-center space-x-2 font-mono text-xs">
        <span className={`px-2.5 py-1 rounded-md border font-semibold ${STATUS_COLORS[status]?.badge || ''}`}>
          LAYOUT {status}
        </span>
        <span className="px-2.5 py-1 rounded-md bg-command-surface/90 border border-command-border text-command-text">
          {drawingMode === 'draw_space'
            ? 'DRAWING PARKING SPACE (Click to add vertices, click start to close)'
            : drawingMode === 'draw_approach'
            ? 'DRAWING APPROACH ZONE (Click points, click start to close)'
            : drawingMode === 'edit_vertices'
            ? 'EDIT VERTICES (Drag cyan/purple handles)'
            : 'INSPECT MODE (Click space to select)'}
        </span>
      </div>
    </div>
  );
}
