'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  Camera,
  CheckCircle2,
  Clock,
  FileImage,
  Info,
  RefreshCw,
  ScanLine,
  ServerOff,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import {
  ANPR_ALLOWED_MIME_TYPES,
  ANPR_MAX_IMAGE_BYTES,
  AnprImageResult,
  FORMAT_STATUS_LABELS,
  FormatStatus,
  PlateResult,
  READ_STATUS_LABELS,
  REASON_CODE_LABELS,
  RESULT_STATUS_LABELS,
  ReadStatus,
  ResultStatus,
  QUALITY_GUIDANCE,
  TEXT_CLEANUP_LABELS,
  PLATE_PATTERN_LABELS,
} from '../../lib/anprTypes';
import { createAnnotatedImage, createReviewReport, downloadBlob } from '../../lib/anprExports';
import { RegistrationCard } from './RegistrationCard';
import { ViewportBox, computeContainViewport } from '../../lib/canvasTransform';

export interface AnprInspectionPanelProps {
  onAnalyze?: (file: File) => Promise<AnprImageResult>;
}

const ALLOWED_EXTENSION_PATTERN = /\.(jpe?g|png)$/i;

function isSupportedImageFile(file: File): boolean {
  const mimeAllowed = (ANPR_ALLOWED_MIME_TYPES as readonly string[]).includes(file.type);
  const extAllowed = ALLOWED_EXTENSION_PATTERN.test(file.name);
  if (file.type) {
    return mimeAllowed;
  }
  return extAllowed;
}

function formatByteSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const kib = bytes / 1024;
  if (kib < 1024) {
    return `${kib.toFixed(1)} KiB`;
  }
  const mib = kib / 1024;
  return `${mib.toFixed(2)} MiB`;
}

function formatScore(score: number | null): string {
  if (score === null || !Number.isFinite(score)) {
    return 'Missing';
  }
  return score.toFixed(2);
}

function getResultStatusStyle(status: ResultStatus): {
  banner: string;
  badge: string;
  description: string;
} {
  switch (status) {
    case 'COMPLETE':
      return {
        banner: 'bg-emerald-50 border-emerald-200 text-emerald-700',
        badge: 'bg-emerald-50 text-emerald-700 border-emerald-200',
        description:
          'All detected plates produced a candidate read matching the supported IN_COMMON_V1 format shape.',
      };
    case 'NEEDS_REVIEW':
      return {
        banner: 'bg-amber-50 border-amber-200 text-amber-700',
        badge: 'bg-amber-50 text-amber-700 border-amber-200',
        description:
          'Check each candidate against its image crop. Review-only engines intentionally require confirmation; other reasons can include uncertain text or unsupported format.',
      };
    case 'NO_PLATE':
      return {
        banner: 'bg-slate-100 border-command-border text-command-text',
        badge: 'bg-command-elevated text-command-muted border-command-border',
        description: 'No license plate bounding boxes were detected in this image.',
      };
    case 'UNAVAILABLE':
      return {
        banner: 'bg-rose-50 border-rose-200 text-rose-700',
        badge: 'bg-rose-50 text-rose-700 border-rose-200',
        description:
          'Local recognition runtime or required weights are unavailable. No plate results were generated.',
      };
  }
}

function getReadStatusStyle(status: ReadStatus): {
  badge: string;
  boxBorder: string;
  boxFill: string;
  boxTag: string;
} {
  switch (status) {
    case 'READ':
      return {
        badge: 'bg-emerald-50 text-emerald-700 border-emerald-200',
        boxBorder: '#22C55E',
        boxFill: 'rgba(34, 197, 94, 0.16)',
        boxTag: 'bg-emerald-50 text-emerald-700 border-emerald-200',
      };
    case 'NEEDS_REVIEW':
      return {
        badge: 'bg-amber-50 text-amber-700 border-amber-200',
        boxBorder: '#F59E0B',
        boxFill: 'rgba(245, 158, 11, 0.16)',
        boxTag: 'bg-amber-50 text-amber-700 border-amber-200',
      };
    case 'UNREADABLE':
      return {
        badge: 'bg-rose-50 text-rose-700 border-rose-200',
        boxBorder: '#EF4444',
        boxFill: 'rgba(239, 68, 68, 0.16)',
        boxTag: 'bg-rose-50 text-rose-700 border-rose-200',
      };
  }
}

function getFormatStatusStyle(status: FormatStatus): string {
  switch (status) {
    case 'MATCHED':
      return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    case 'NO_MATCH':
      return 'bg-amber-50 text-amber-700 border-amber-200';
    case 'EMPTY':
      return 'bg-slate-100 text-command-muted border-command-border';
  }
}

export function AnprInspectionPanel({ onAnalyze }: AnprInspectionPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [naturalDimensions, setNaturalDimensions] = useState<{ width: number; height: number } | null>(
    null
  );
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AnprImageResult | null>(null);
  const [selectedPlateId, setSelectedPlateId] = useState<string | null>(null);
  const [viewport, setViewport] = useState<ViewportBox>({ x0: 0, y0: 0, width: 0, height: 0 });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const viewportContainerRef = useRef<HTMLDivElement>(null);
  const activeObjectUrlRef = useRef<string | null>(null);
  const requestCounterRef = useRef(0);

  const revokeCurrentObjectUrl = useCallback(() => {
    if (activeObjectUrlRef.current) {
      URL.revokeObjectURL(activeObjectUrlRef.current);
      activeObjectUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      revokeCurrentObjectUrl();
      requestCounterRef.current += 1;
    };
  }, [revokeCurrentObjectUrl]);

  const referenceWidth = result?.image_width ?? naturalDimensions?.width ?? 0;
  const referenceHeight = result?.image_height ?? naturalDimensions?.height ?? 0;

  const updateViewport = useCallback(() => {
    const container = viewportContainerRef.current;
    if (!container || referenceWidth <= 0 || referenceHeight <= 0) {
      setViewport({ x0: 0, y0: 0, width: 0, height: 0 });
      return;
    }
    const cssW = container.clientWidth;
    const cssH = container.clientHeight;
    if (cssW <= 0 || cssH <= 0) {
      setViewport({ x0: 0, y0: 0, width: 0, height: 0 });
      return;
    }
    setViewport(computeContainViewport(cssW, cssH, referenceWidth, referenceHeight));
  }, [referenceWidth, referenceHeight]);

  useEffect(() => {
    updateViewport();
  }, [updateViewport, previewUrl, result]);

  useEffect(() => {
    const container = viewportContainerRef.current;
    if (!container || typeof ResizeObserver === 'undefined') {
      return;
    }
    const observer = new ResizeObserver(() => {
      updateViewport();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [updateViewport]);

  const applySelectedFile = useCallback(
    (file: File | null) => {
      // Invalidate any in-flight recognition call and clear stale results immediately.
      requestCounterRef.current += 1;
      setIsAnalyzing(false);
      setIsExporting(false);
      setResult(null);
      setSelectedPlateId(null);
      setNaturalDimensions(null);

      revokeCurrentObjectUrl();

      if (!file) {
        setSelectedFile(null);
        setPreviewUrl(null);
        setErrorMessage(null);
        return;
      }

      if (!isSupportedImageFile(file)) {
        setSelectedFile(null);
        setPreviewUrl(null);
        setErrorMessage('Unsupported file format. Please select a single JPEG or PNG image.');
        return;
      }

      if (file.size <= 0) {
        setSelectedFile(null);
        setPreviewUrl(null);
        setErrorMessage('Selected image file is empty.');
        return;
      }

      if (file.size > ANPR_MAX_IMAGE_BYTES) {
        setSelectedFile(null);
        setPreviewUrl(null);
        setErrorMessage(
          `Image exceeds the 10 MiB limit (${formatByteSize(file.size)}). Choose a JPEG or PNG up to 10 MiB.`
        );
        return;
      }

      const nextUrl = URL.createObjectURL(file);
      activeObjectUrlRef.current = nextUrl;
      setSelectedFile(file);
      setPreviewUrl(nextUrl);
      setErrorMessage(null);
    },
    [revokeCurrentObjectUrl]
  );

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) {
      return;
    }
    if (files.length > 1) {
      applySelectedFile(null);
      setErrorMessage('Please select only one JPEG or PNG image at a time.');
      e.target.value = '';
      return;
    }
    applySelectedFile(files[0]);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (!files || files.length === 0) {
      return;
    }
    if (files.length > 1) {
      applySelectedFile(null);
      setErrorMessage('Please drop only one JPEG or PNG image at a time.');
      return;
    }
    applySelectedFile(files[0]);
  };

  const handleClearImage = () => {
    applySelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyze = async () => {
    if (!onAnalyze || !selectedFile || isAnalyzing) {
      return;
    }

    const requestId = requestCounterRef.current + 1;
    requestCounterRef.current = requestId;
    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const response = await onAnalyze(selectedFile);
      if (requestCounterRef.current !== requestId) {
        return;
      }
      setResult(response);
      setSelectedPlateId(response.plates.length > 0 ? response.plates[0].plate_id : null);
    } catch (err: unknown) {
      if (requestCounterRef.current !== requestId) {
        return;
      }
      const message =
        err instanceof Error && err.message
          ? err.message
          : 'Plate recognition failed. Check the local recognition service and try again.';
      setResult(null);
      setSelectedPlateId(null);
      setErrorMessage(message);
    } finally {
      if (requestCounterRef.current === requestId) {
        setIsAnalyzing(false);
      }
    }
  };

  const isServiceConnected = typeof onAnalyze === 'function';
  const canRunRecognition = Boolean(isServiceConnected && selectedFile && !isAnalyzing && !isExporting);

  const handleAnnotatedExport = async () => {
    if (!result || !previewUrl || isExporting) return;
    const requestId = requestCounterRef.current;
    setIsExporting(true);
    try {
      const blob = await createAnnotatedImage(previewUrl, result);
      if (requestCounterRef.current === requestId) {
        downloadBlob(blob, `roadsense-anpr-${result.request_id.replace(/[^a-zA-Z0-9_-]/g, '')}.png`);
      }
    } catch (err) {
      if (requestCounterRef.current === requestId) {
        setErrorMessage(err instanceof Error ? err.message : 'Annotated export failed.');
      }
    } finally {
      if (requestCounterRef.current === requestId) setIsExporting(false);
    }
  };

  return (
    <section
      aria-label="ANPR Image Inspection"
      className="space-y-5 font-sans text-xs text-command-text"
    >
      {/* Brand hero: what the tool does, with honestly labelled measured results */}
      <div className="relative overflow-hidden rounded-4xl bg-command-surface border border-command-border shadow-subtle px-7 py-9 lg:px-12 lg:py-12">
        <div className="absolute inset-0 hero-glow pointer-events-none" aria-hidden="true" />
        <div className="relative grid grid-cols-1 lg:grid-cols-12 gap-8 items-end reveal">
          <div className="lg:col-span-7 space-y-5">
            <p className="inline-flex items-center gap-2 rounded-full bg-brand-soft px-3 py-1 text-[12px] font-medium text-brand-deep">
              <span className="h-1.5 w-1.5 rounded-full bg-brand" aria-hidden="true" />
              Number plate recognition · India
            </p>
            <h1 className="text-[40px] leading-[1.04] lg:text-[56px] font-semibold tracking-[-0.035em] text-command-text">
              Find the plate.
              <br />
              Read it. Check it.
              <br />
              <span className="text-brand">Let a person decide.</span>
            </h1>
            <p className="max-w-xl text-[15px] leading-relaxed text-command-muted">
              A trained detector finds the plate, OCR reads it, the format and issuing RTO are checked, and every
              uncertain reading goes to review with a reason. Everything runs on this machine.
            </p>
          </div>
          <dl className="lg:col-span-5 grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-1 gap-3">
            {[
              { value: '58 / 94', label: 'validation plates read exactly' },
              { value: '33 · 0 wrong', label: 'auto-accepted with the opt-in policy' },
              { value: '1,302', label: 'RTO codes decoded offline' },
            ].map((stat) => (
              <div key={stat.label} className="flex flex-col-reverse rounded-3xl bg-command-bg border border-command-border px-5 py-4">
                <dt className="mt-1.5 text-[12px] leading-snug text-command-muted">{stat.label}</dt>
                <dd className="text-[26px] font-semibold tracking-[-0.02em] leading-none text-command-text">{stat.value}</dd>
              </div>
            ))}
          </dl>
        </div>
        <p className="relative mt-8 text-[11px] text-command-muted">
          Validation results on mostly truck images with publisher labels; the held-out test is pending. Not a
          registration lookup or toll system.
        </p>
      </div>

      {/* Header & Controls Bar */}
      <div id="anpr-workspace" className="scroll-mt-24 p-6 rounded-xl glass-panel border border-command-border space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-command-border pb-3">
          <div className="flex items-center space-x-2">
            <Camera className="w-5 h-5 text-brand shrink-0" />
            <div>
              <h2 className="text-[22px] leading-tight font-medium text-command-text">
                Inspect an image
              </h2>
              <p className="mt-1 text-[13px] leading-snug text-command-muted">
                Plate detection, raw and normalized OCR, format check and issuing office
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className="px-3.5 py-1.5 rounded-full bg-command-elevated border border-command-border text-[12px] text-command-muted">
              Profile <strong className="font-medium text-command-text">IN_COMMON_V1</strong>
            </span>
            <span className="px-3.5 py-1.5 rounded-full bg-command-elevated border border-command-border text-[12px] text-command-muted">
              JPEG / PNG · max 10 MiB
            </span>
          </div>
        </div>

        {/* Service Connection Notice when onAnalyze is omitted */}
        {!isServiceConnected && (
          <div
            role="status"
            aria-live="polite"
            className="flex items-center space-x-2 p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 text-xs"
          >
            <ServerOff className="w-4 h-4 text-amber-600 shrink-0" />
            <span>Recognition service is not connected yet.</span>
          </div>
        )}

        {/* File Picker & Action Row */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-stretch">
          <div
            role="button"
            tabIndex={0}
            aria-label="Select a JPEG or PNG vehicle image up to 10 MiB"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            className={`lg:col-span-8 border-2 border-dashed rounded-3xl p-5 flex items-center justify-between gap-3 cursor-pointer transition-all focus:outline-none focus:ring-2 focus:ring-brand ${
              selectedFile
                ? 'border-brand/70 bg-brand-soft'
                : 'border-command-border hover:border-command-subtle bg-command-bg/60'
            }`}
          >
            <input
              ref={fileInputRef}
              id="anpr-image-upload-input"
              type="file"
              accept="image/jpeg,image/png,.jpg,.jpeg,.png"
              onChange={handleFileInputChange}
              disabled={isAnalyzing}
              className="hidden"
            />

            <div className="flex items-center space-x-3 min-w-0">
              {selectedFile ? (
                <FileImage className="w-6 h-6 text-brand shrink-0" />
              ) : (
                <UploadCloud className="w-6 h-6 text-command-muted shrink-0" />
              )}
              <div className="min-w-0">
                {selectedFile ? (
                  <>
                    <div className="text-[14px] font-medium text-command-text truncate">
                      {selectedFile.name}
                    </div>
                    <div className="text-[11px] text-command-muted">
                      {formatByteSize(selectedFile.size)}
                      {naturalDimensions
                        ? ` • ${naturalDimensions.width}×${naturalDimensions.height} px`
                        : ''}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-[14px] font-medium text-command-text">
                      Drop a vehicle or gate image here, or{' '}
                      <span className="text-brand underline underline-offset-2">browse</span>
                    </div>
                    <div className="text-[11px] text-command-muted">
                      Single JPEG or PNG image up to 10 MiB (local preview only)
                    </div>
                  </>
                )}
              </div>
            </div>

            {selectedFile && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleClearImage();
                }}
                disabled={isAnalyzing}
                className="flex items-center space-x-1.5 px-3.5 py-2 rounded-full bg-command-elevated hover:bg-command-border border border-command-border text-command-muted hover:text-command-text text-[12px] shrink-0 transition-colors disabled:opacity-40"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear</span>
              </button>
            )}
          </div>

          <div className="lg:col-span-4 flex flex-col justify-center">
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={!canRunRecognition}
              aria-busy={isAnalyzing}
              className="w-full h-full min-h-[56px] flex items-center justify-center space-x-2 px-6 py-3 rounded-full bg-brand hover:bg-brand-bright text-white font-medium text-[15px] shadow-brand transition-colors disabled:opacity-40 disabled:shadow-none disabled:cursor-not-allowed"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin shrink-0" />
                  <span>Inspecting plate…</span>
                </>
              ) : (
                <>
                  <ScanLine className="w-4 h-4 shrink-0" />
                  <span>Run plate recognition</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Validation or Runtime Error Alert */}
        {errorMessage && (
          <div
            role="alert"
            className="flex items-start space-x-2 p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs"
          >
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <span>{errorMessage}</span>
          </div>
        )}
      </div>

      {result && !isAnalyzing && (
        <div className="flex flex-wrap items-center gap-3 p-4 rounded-xl glass-panel border border-command-border">
          <button type="button" onClick={handleAnnotatedExport} disabled={!previewUrl || isExporting}
            className="px-5 py-2.5 rounded-full bg-command-text text-white font-medium text-[13px] hover:bg-slate-700 transition-colors disabled:opacity-40">
            {isExporting ? 'Preparing image…' : 'Download annotated PNG'}
          </button>
          <button type="button" disabled={isExporting} onClick={() => downloadBlob(createReviewReport(result), `roadsense-anpr-${result.request_id.replace(/[^a-zA-Z0-9_-]/g, '')}.json`)}
            className="px-5 py-2.5 rounded-full bg-command-elevated border border-command-border text-command-text text-[13px] hover:bg-command-border transition-colors disabled:opacity-40">
            Download review JSON
          </button>
          <span className="font-sans text-command-muted">Exports stay in your browser. Readings are not human verified.</span>
        </div>
      )}

      {/* Main Split Grid: Image Viewport + Plate Inspection Details */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Local Preview & Bounding Box Overlay */}
        <div className="lg:col-span-7 p-6 rounded-xl glass-panel border border-command-border flex flex-col space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-[17px] leading-snug font-medium text-command-text">
              Image and detected plates
            </span>
            {result && (
              <span className="text-[11px] text-command-muted">
                Normalized Frame: {result.image_width}×{result.image_height} px •{' '}
                {result.plates.length} {result.plates.length === 1 ? 'plate' : 'plates'}
              </span>
            )}
          </div>

          <div
            ref={viewportContainerRef}
            className="relative w-full h-[380px] sm:h-[440px] rounded-lg bg-[#0B0F12] border border-command-border overflow-hidden select-none"
          >
            {previewUrl ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt={selectedFile ? `Preview of ${selectedFile.name}` : 'Selected vehicle preview'}
                  onLoad={(e) => {
                    const img = e.currentTarget;
                    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                      setNaturalDimensions({
                        width: img.naturalWidth,
                        height: img.naturalHeight,
                      });
                    }
                  }}
                  onError={() => {
                    setErrorMessage('Unable to decode the selected image file for preview.');
                  }}
                  style={
                    viewport.width > 0 && viewport.height > 0
                      ? {
                          position: 'absolute',
                          left: `${viewport.x0}px`,
                          top: `${viewport.y0}px`,
                          width: `${viewport.width}px`,
                          height: `${viewport.height}px`,
                        }
                      : undefined
                  }
                  className={
                    viewport.width > 0 && viewport.height > 0
                      ? 'block pointer-events-none'
                      : 'w-full h-full object-contain pointer-events-none'
                  }
                />

                {/* Bounding Box Overlay Layer */}
                {result &&
                  result.plates.length > 0 &&
                  viewport.width > 0 &&
                  viewport.height > 0 &&
                  result.image_width > 0 &&
                  result.image_height > 0 && (
                    <div
                      aria-label="Detected plate bounding boxes"
                      style={{
                        position: 'absolute',
                        left: `${viewport.x0}px`,
                        top: `${viewport.y0}px`,
                        width: `${viewport.width}px`,
                        height: `${viewport.height}px`,
                      }}
                    >
                      {result.plates.map((plate, index) => {
                        const scaleX = viewport.width / result.image_width;
                        const scaleY = viewport.height / result.image_height;
                        const left = plate.bbox.x1 * scaleX;
                        const top = plate.bbox.y1 * scaleY;
                        const width = Math.max(2, (plate.bbox.x2 - plate.bbox.x1) * scaleX);
                        const height = Math.max(2, (plate.bbox.y2 - plate.bbox.y1) * scaleY);
                        const readStyle = getReadStatusStyle(plate.recognition_status);
                        const isSelected = plate.plate_id === selectedPlateId;
                        const displayLabel =
                          plate.candidate_text || plate.normalized_text || READ_STATUS_LABELS[plate.recognition_status];

                        return (
                          <button
                            key={plate.plate_id}
                            type="button"
                            onClick={() => setSelectedPlateId(plate.plate_id)}
                            aria-label={`Plate ${index + 1} (${plate.plate_id}): ${displayLabel}`}
                            style={{
                              position: 'absolute',
                              left: `${left}px`,
                              top: `${top}px`,
                              width: `${width}px`,
                              height: `${height}px`,
                              borderColor: readStyle.boxBorder,
                              backgroundColor: readStyle.boxFill,
                              borderWidth: isSelected ? '2.5px' : '1.75px',
                            }}
                            className="group rounded-sm border transition-shadow focus:outline-none focus:ring-2 focus:ring-accent-cyan"
                          >
                            <span
                              className={`absolute left-0 -top-6 px-1.5 py-0.5 rounded border text-[10px] font-bold whitespace-nowrap shadow ${readStyle.boxTag}`}
                            >
                              #{index + 1} {displayLabel}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  )}
              </>
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-center p-6 text-command-muted space-y-2">
                <FileImage className="w-10 h-10 opacity-40" />
                <p className="text-xs font-medium text-command-text">No vehicle image selected</p>
                <p className="text-[11px] font-sans max-w-sm">
                  Select a close-range vehicle or gate-camera JPEG/PNG image to preview it and
                  inspect detected license plate bounds.
                </p>
              </div>
            )}
          </div>

          <div className="flex items-start space-x-2 text-[11px] font-sans text-command-muted bg-command-bg/70 p-2.5 rounded-lg border border-command-border">
            <Info className="w-4 h-4 text-accent-cyan shrink-0 mt-0.5" />
            <span>
              Plate boxes use EXIF-normalized source-image pixel coordinates scaled to the displayed
              contain viewport. Format checks test only the <code className="font-mono">IN_COMMON_V1</code>{' '}
              character pattern and do not verify official vehicle registration records.
            </span>
          </div>
        </div>

        {/* Right: Result Status & Per-Plate OCR Readouts */}
        <div className="lg:col-span-5 p-6 rounded-xl glass-panel border border-command-border flex flex-col space-y-4">
          <div className="flex items-center justify-between border-b border-command-border pb-3">
            <span className="text-[17px] leading-snug font-medium text-command-text">
              Reading and checks
            </span>
            {result && (
              <span className="flex items-center space-x-1 text-[11px] text-command-muted">
                <Clock className="w-3.5 h-3.5" />
                <span>{result.processing_ms.toFixed(1)} ms</span>
              </span>
            )}
          </div>

          {!result ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-6 rounded-lg bg-command-bg/50 border border-command-border text-command-muted space-y-2 min-h-[260px]">
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-7 h-7 text-radar-bright animate-spin" />
                  <p className="text-xs font-semibold text-command-text">
                    Running plate detection and OCR...
                  </p>
                  <p className="text-[11px] font-sans">
                    Evaluating candidate plate crops and checking IN_COMMON_V1 format shape.
                  </p>
                </>
              ) : (
                <>
                  <ScanLine className="w-7 h-7 opacity-40" />
                  <p className="text-xs font-semibold text-command-text">
                    No recognition result to display
                  </p>
                  <p className="text-[11px] font-sans max-w-xs">
                    {!isServiceConnected
                      ? 'Recognition service is not connected yet.'
                      : selectedFile
                      ? 'Select "Run plate recognition" to inspect the selected image.'
                      : 'Choose a JPEG or PNG image to begin inspection.'}
                  </p>
                </>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {/* Result Status Banner (COMPLETE, NEEDS_REVIEW, NO_PLATE, UNAVAILABLE) */}
              {(() => {
                const statusStyle = getResultStatusStyle(result.status);
                return (
                  <div className={`p-3 rounded-lg border space-y-1.5 ${statusStyle.banner}`}>
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center space-x-2">
                        {result.status === 'COMPLETE' ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 shrink-0" />
                        )}
                        <span className="font-bold uppercase tracking-wider text-xs">
                          {RESULT_STATUS_LABELS[result.status]}
                        </span>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded border text-[10px] font-bold uppercase ${statusStyle.badge}`}
                      >
                        {result.status}
                      </span>
                    </div>
                    <p className="text-[11px] font-sans opacity-90">{statusStyle.description}</p>
                    <div className="text-[10px] opacity-75 pt-0.5">
                      Request ID: <span className="font-semibold">{result.request_id}</span> • Schema
                      v{result.schema_version}
                    </div>
                  </div>
                );
              })()}

              {/* Top-level Warnings */}
              {result.warnings.length > 0 && (
                <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 space-y-1">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-amber-700">
                    Inspection Warnings
                  </div>
                  <ul className="list-disc list-inside text-[11px] font-sans space-y-0.5">
                    {result.warnings.map((warning, idx) => (
                      <li key={idx}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Empty-Plate States for NO_PLATE / UNAVAILABLE */}
              {result.plates.length === 0 ? (
                <div className="p-4 rounded-lg bg-command-bg border border-command-border text-center text-command-muted text-xs font-sans">
                  {result.status === 'UNAVAILABLE'
                    ? 'Recognition runtime is unavailable. No plate detections or OCR scores were produced.'
                    : 'No plate detections were returned for this image.'}
                </div>
              ) : (
                <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
                  {result.plates.map((plate: PlateResult, index: number) => {
                    const readStyle = getReadStatusStyle(plate.recognition_status);
                    const formatBadge = getFormatStatusStyle(plate.format_status);
                    const isSelected = plate.plate_id === selectedPlateId;

                    return (
                      <div
                        key={plate.plate_id}
                        onClick={() => setSelectedPlateId(plate.plate_id)}
                        className={`p-3 rounded-lg border transition-colors cursor-pointer space-y-2.5 ${
                          isSelected
                            ? 'bg-command-elevated/90 border-radar-bright'
                            : 'bg-command-bg border-command-border hover:border-command-subtle'
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-command-border pb-2">
                          <div className="flex items-center space-x-2">
                            <span className="px-1.5 py-0.5 rounded bg-command-surface border border-command-border text-[10px] font-bold text-command-text">
                              #{index + 1}
                            </span>
                            <span className="font-bold text-command-text">{plate.plate_id}</span>
                          </div>
                          <span
                            className={`px-2 py-0.5 rounded border text-[10px] font-bold uppercase ${readStyle.badge}`}
                          >
                            {READ_STATUS_LABELS[plate.recognition_status]} ({plate.recognition_status})
                          </span>
                        </div>

                        {/* Raw vs Normalized Text */}
                        {isSelected && previewUrl && (
                          <div className="rounded border border-command-border overflow-hidden bg-command-surface">
                            <div className="p-2 text-command-muted text-[10px]">Detected crop — enlarged for inspection, not restored detail</div>
                            <svg role="img" aria-label={`Enlarged crop for plate ${index + 1}`}
                              viewBox={`${plate.bbox.x1} ${plate.bbox.y1} ${plate.bbox.x2 - plate.bbox.x1} ${plate.bbox.y2 - plate.bbox.y1}`}
                              className="w-full h-32" preserveAspectRatio="xMidYMid meet">
                              <image href={previewUrl} width={result.image_width} height={result.image_height} />
                            </svg>
                          </div>
                        )}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          <div className="p-2 rounded bg-command-surface/80 border border-command-border">
                            <span className="block text-[10px] uppercase text-command-muted">
                              Raw OCR Text
                            </span>
                            <div className="mt-0.5 text-xs font-bold text-command-text break-all">
                              {plate.raw_text ? (
                                <span className="whitespace-pre-wrap">{plate.raw_text}</span>
                              ) : (
                                <span className="italic text-command-muted">Empty</span>
                              )}
                            </div>
                          </div>

                          <div className="p-2 rounded bg-command-surface/80 border border-command-border">
                            <span className="block text-[10px] uppercase text-command-muted">
                              Normalized Text
                            </span>
                            <div className="mt-0.5 text-xs font-bold text-radar-bright tracking-wider break-all">
                              {plate.normalized_text ? (
                                <span>{plate.normalized_text}</span>
                              ) : (
                                <span className="italic text-command-muted font-normal">Empty</span>
                              )}
                            </div>
                          </div>
                        </div>

                        {plate.plate_format && (
                          <div className="p-2 rounded bg-command-surface/60 border border-command-border space-y-1">
                            <span className="block text-[10px] uppercase text-command-muted">
                              Plate format check ({plate.plate_format.profile}, {plate.plate_format.assessed_text_source} text)
                            </span>
                            <div className="text-[11px] text-command-text">
                              {plate.plate_format.pattern
                                ? PLATE_PATTERN_LABELS[plate.plate_format.pattern]
                                : 'No supported pattern'}
                              {plate.plate_format.state_code && (
                                <span className="ml-2">
                                  State code {plate.plate_format.state_code}:{' '}
                                  {plate.plate_format.state_code_known ? 'known' : 'not a known code'}
                                </span>
                              )}
                            </div>
                            {plate.plate_format.lookalike_suggestions.length > 0 && (
                              <div className="text-[11px] text-command-muted">
                                Possible readings (look-alike hints, not applied):{' '}
                                <span className="font-bold tracking-wider text-command-text">
                                  {plate.plate_format.lookalike_suggestions.join(', ')}
                                </span>
                              </div>
                            )}
                            {plate.plate_format.registration_origin && (
                              <div className="text-[11px] text-command-text">
                                Issued by:{' '}
                                <span className="font-bold">
                                  {plate.plate_format.registration_origin.state_name ?? plate.plate_format.registration_origin.state_code}
                                  {' · '}
                                  {plate.plate_format.registration_origin.rto_code}
                                  {plate.plate_format.registration_origin.rto_office
                                    ? ` · ${plate.plate_format.registration_origin.rto_office}`
                                    : ''}
                                </span>
                                {!plate.plate_format.registration_origin.rto_in_table && (
                                  <span className="ml-1 text-command-muted">(RTO not in table)</span>
                                )}
                                {plate.plate_format.registration_origin.rto_jurisdiction && (
                                  <div className="text-[10px] text-command-muted">
                                    {plate.plate_format.registration_origin.rto_jurisdiction}
                                  </div>
                                )}
                                {plate.plate_format.registration_origin.rto_notes && (
                                  <div className="text-[10px] text-command-muted">
                                    {plate.plate_format.registration_origin.rto_notes}
                                  </div>
                                )}
                              </div>
                            )}
                            <p className="text-[10px] text-command-muted">
                              A supported pattern or known state code does not verify a registration. Issuing office
                              comes from a public RTO directory (Wikipedia rev{' '}
                              {plate.plate_format.registration_origin?.source_revision ?? 'n/a'}, CC BY-SA 4.0); it is
                              not a vehicle or owner record and is decoded from the unverified OCR text.
                            </p>
                          </div>
                        )}

                        {plate.candidate_text && (
                          <div className="p-2 rounded bg-command-surface/80 border border-command-border">
                            <span className="block text-[10px] uppercase text-command-muted">
                              Review Candidate (cleaned, not verified)
                            </span>
                            <div className="mt-0.5 text-xs font-bold text-radar-bright tracking-wider break-all">
                              {plate.candidate_text}
                            </div>
                            <ul className="mt-1 text-[10px] text-command-muted list-disc list-inside">
                              {(plate.text_cleanup ?? []).map((step) => (
                                <li key={step}>{TEXT_CLEANUP_LABELS[step] ?? step}</li>
                              ))}
                              {plate.candidate_format_status && (
                                <li>Candidate format: {FORMAT_STATUS_LABELS[plate.candidate_format_status]}</li>
                              )}
                            </ul>
                          </div>
                        )}

                        {/* Format Status & Uncalibrated Scores */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          <div className="p-2 rounded bg-command-surface/60 border border-command-border space-y-1">
                            <span className="block text-[10px] uppercase text-command-muted">
                              Format Status ({plate.format_profile})
                            </span>
                            <span
                              className={`inline-block px-2 py-0.5 rounded border text-[10px] font-bold ${formatBadge}`}
                            >
                              {FORMAT_STATUS_LABELS[plate.format_status]}
                            </span>
                          </div>

                          <div className="p-2 rounded bg-command-surface/60 border border-command-border space-y-0.5">
                            <span className="block text-[10px] uppercase text-command-muted">
                              OCR Score (Uncalibrated)
                            </span>
                            <div className="text-xs font-bold text-command-text">
                              {plate.reason_codes.includes('OCR_REVIEW_ONLY') ? 'Not used — human review required' : formatScore(plate.ocr_confidence)}
                            </div>
                            <div className="text-[10px] text-command-muted">
                              Detector score: {formatScore(plate.detection_confidence)}
                            </div>
                          </div>
                        </div>

                        {plate.image_quality && (
                          <div className="p-2 rounded border border-command-border bg-command-surface/60 space-y-1 font-sans">
                            <div className="font-bold text-command-text">Capture quality and retake guidance</div>
                            <div className="text-command-muted text-[10px]">
                              Detail score: {plate.image_quality.sharpness_score.toFixed(1)} • Brightness: {plate.image_quality.mean_brightness.toFixed(0)}/255
                              {' • '}2D line tilt: {plate.image_quality.observed_line_tilt_degrees === null ? 'insufficient evidence' : `${plate.image_quality.observed_line_tilt_degrees.toFixed(1)}°`}
                            </div>
                            {plate.image_quality.warnings.map(flag => (
                              <p key={flag} className="text-amber-700">{QUALITY_GUIDANCE[flag]}</p>
                            ))}
                            <p className="text-command-muted text-[10px]">
                              Advisory measurements only. {plate.image_quality.warnings.length === 0 ? 'No configured cues triggered; this does not verify the text.' : 'Check the crop before using any reading.'}
                            </p>
                          </div>
                        )}

                        {/* Bounding Box Coordinates */}
                        <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-command-muted pt-0.5">
                          <span>
                            Box: ({Math.round(plate.bbox.x1)}, {Math.round(plate.bbox.y1)}) → (
                            {Math.round(plate.bbox.x2)}, {Math.round(plate.bbox.y2)}) px
                          </span>
                          <span>
                            Size: {Math.round(plate.bbox.x2 - plate.bbox.x1)}×
                            {Math.round(plate.bbox.y2 - plate.bbox.y1)} px
                          </span>
                        </div>

                        {/* Review Reason Codes */}
                        {plate.reason_codes.length > 0 && (
                          <div className="pt-1.5 border-t border-command-border space-y-1">
                            <span className="block text-[10px] uppercase text-amber-700 font-bold">
                              Review Reason Codes
                            </span>
                            <ul className="space-y-1">
                              {plate.reason_codes.map((code) => (
                                <li
                                  key={code}
                                  className="flex items-start space-x-1.5 text-[11px] font-sans text-amber-700"
                                >
                                  <span className="font-mono text-[10px] px-1.5 py-0.2 rounded bg-amber-50 border border-amber-200 text-amber-700 shrink-0">
                                    {code}
                                  </span>
                                  <span>{REASON_CODE_LABELS[code] ?? code}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Registration card for the selected plate: real issuing office + simulated demo owner section */}
      {result && !isAnalyzing && result.plates.length > 0 && (
        <RegistrationCard
          plate={result.plates.find((p) => p.plate_id === selectedPlateId) ?? result.plates[0]}
        />
      )}
    </section>
  );
}
