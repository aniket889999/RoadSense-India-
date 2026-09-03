'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  SkipBack,
  Sliders,
  Volume2,
  Maximize,
  Sparkles,
} from 'lucide-react';
import { DriveSession, RawDetection, RoadEvent } from '../lib/types';
import { getArtifactDownloadUrl } from '../lib/api';

interface VideoViewportProps {
  session: DriveSession;
  detections: RawDetection[];
  roadEvents: RoadEvent[];
  selectedEvent: RoadEvent | null;
  onSelectEvent: (event: RoadEvent) => void;
  confidenceFilter: number;
  onConfidenceFilterChange: (val: number) => void;
}

export function VideoViewport({
  session,
  detections,
  roadEvents,
  selectedEvent,
  onSelectEvent,
  confidenceFilter,
  onConfidenceFilterChange,
}: VideoViewportProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(session.source_duration_seconds || 0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [visibleDetectionsCount, setVisibleDetectionsCount] = useState(0);

  const videoUrl = getArtifactDownloadUrl(session.id, 'raw_video');

  // Jump to selected event when reviewer clicks on an item in the queue
  useEffect(() => {
    if (selectedEvent && videoRef.current) {
      videoRef.current.currentTime = selectedEvent.first_seen_seconds;
      setCurrentTime(selectedEvent.first_seen_seconds);
    }
  }, [selectedEvent]);

  // Synchronized Canvas Overlay Drawing
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const renderOverlay = () => {
      // Match canvas dimensions to video render size
      const width = video.clientWidth;
      const height = video.clientHeight;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.clearRect(0, 0, width, height);

      const sourceWidth = session.source_width || 1280;
      const sourceHeight = session.source_height || 720;
      const scaleX = width / sourceWidth;
      const scaleY = height / sourceHeight;

      const curTime = video.currentTime;
      // Detections active within +/- 0.15s window of current playback frame
      const activeDetections = detections.filter(
        (d) =>
          Math.abs(d.timestamp_seconds - curTime) <= 0.15 &&
          d.confidence >= confidenceFilter
      );

      setVisibleDetectionsCount(activeDetections.length);

      for (const det of activeDetections) {
        const x_min = det.x_min * scaleX;
        const y_min = det.y_min * scaleY;
        const x_max = det.x_max * scaleX;
        const y_max = det.y_max * scaleY;

        const centerX = (x_min + x_max) / 2;
        const centerY = (y_min + y_max) / 2;
        const radius = Math.max(12, Math.max(x_max - x_min, y_max - y_min) / 2 + 6);

        // Radar green target circle
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.stroke();

        // Pulsing glow circle
        ctx.strokeStyle = 'rgba(34, 197, 94, 0.35)';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius + 2, 0, 2 * Math.PI);
        ctx.stroke();

        // Target center crosshair
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(centerX - 5, centerY);
        ctx.lineTo(centerX + 5, centerY);
        ctx.moveTo(centerX, centerY - 5);
        ctx.lineTo(centerX, centerY + 5);
        ctx.stroke();

        // Label pill
        const labelText = `POTHOLE SUGGESTION ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 11px ui-monospace, SFMono-Regular, monospace';
        const textMetrics = ctx.measureText(labelText);
        const textWidth = textMetrics.width;

        const labelX = Math.max(8, Math.min(width - textWidth - 16, x_min));
        const labelY = Math.max(22, y_min - 8);

        ctx.fillStyle = 'rgba(11, 15, 20, 0.85)';
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(labelX - 4, labelY - 14, textWidth + 8, 18, 4);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#22C55E';
        ctx.fillText(labelText, labelX, labelY);
      }
    };

    let animationFrameId: number;
    const loop = () => {
      renderOverlay();
      animationFrameId = requestAnimationFrame(loop);
    };
    loop();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [session, detections, confidenceFilter]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
      if (!duration && videoRef.current.duration) {
        setDuration(videoRef.current.duration);
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const handleStep = (seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + seconds));
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const cycleSpeed = () => {
    const speeds = [0.25, 0.5, 1.0, 1.5, 2.0];
    const nextIdx = (speeds.indexOf(playbackRate) + 1) % speeds.length;
    const nextSpeed = speeds[nextIdx];
    setPlaybackRate(nextSpeed);
    if (videoRef.current) {
      videoRef.current.playbackRate = nextSpeed;
    }
  };

  return (
    <div className="flex flex-col h-full rounded-xl glass-panel-elevated border border-command-border overflow-hidden">
      {/* Video Viewport Header Banner */}
      <div className="px-4 py-2 bg-command-surface border-b border-command-border flex items-center justify-between text-xs font-mono">
        <div className="flex items-center space-x-3">
          <span className="text-command-muted uppercase tracking-wider">Viewport:</span>
          <span className="text-command-text font-bold truncate max-w-xs">{session.source_filename}</span>
          <span className="px-2 py-0.5 rounded bg-radar-dim/30 text-radar-bright border border-radar-green/30 text-[10px]">
            DRIVE REVIEW SAMPLED PLAYBACK
          </span>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1 text-command-muted">
            <span>Detections in view:</span>
            <span className="text-radar-bright font-bold">{visibleDetectionsCount}</span>
          </div>
          <div className="flex items-center space-x-1 text-command-muted">
            <span>Time:</span>
            <span className="text-command-text font-bold">
              {currentTime.toFixed(2)}s / {(duration || session.source_duration_seconds || 0).toFixed(2)}s
            </span>
          </div>
        </div>
      </div>

      {/* Synchronized Video & Canvas Stage */}
      <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden min-h-[360px]">
        <video
          ref={videoRef}
          src={videoUrl}
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onLoadedMetadata={() => {
            if (videoRef.current) setDuration(videoRef.current.duration);
          }}
          className="w-full h-full object-contain max-h-[520px]"
          playsInline
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Persistent Non-Claim Safety Watermark */}
        <div className="absolute top-3 left-3 bg-black/80 backdrop-blur-md px-3 py-1.5 rounded border border-command-border text-[10px] font-mono text-command-muted space-y-0.5">
          <div className="text-radar-bright font-semibold">EXPERIMENTAL OPERATOR REVIEW</div>
          <div>GREEN CIRCLES = UNVERIFIED MODEL SUGGESTIONS</div>
        </div>
      </div>

      {/* Timeline with Event Markers */}
      <div className="p-3 bg-command-surface border-t border-command-border space-y-2">
        <div className="relative flex items-center">
          <input
            type="range"
            min="0"
            max={duration || session.source_duration_seconds || 100}
            step="0.05"
            value={currentTime}
            onChange={handleSeek}
            className="w-full h-2 bg-command-elevated rounded-lg appearance-none cursor-pointer accent-radar-bright"
          />

          {/* Render Timeline Event Markers */}
          {duration > 0 &&
            roadEvents.map((ev) => {
              const leftPercent = (ev.first_seen_seconds / duration) * 100;
              const isConfirmed = ev.review_status === 'CONFIRMED';
              const isRejected = ev.review_status === 'REJECTED';
              const markerColor = isConfirmed
                ? 'bg-radar-bright'
                : isRejected
                ? 'bg-accent-red'
                : 'bg-accent-amber';

              return (
                <button
                  key={ev.id}
                  onClick={() => onSelectEvent(ev)}
                  style={{ left: `${Math.min(98, Math.max(2, leftPercent))}%` }}
                  className={`absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full ${markerColor} border border-black shadow hover:scale-150 transition-transform`}
                  title={`Road Event @ ${ev.first_seen_seconds.toFixed(1)}s (${ev.review_status})`}
                />
              );
            })}
        </div>

        {/* Playback Controls & Confidence Filter Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono pt-1">
          {/* Left Media Buttons */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleStep(-1.0)}
              className="p-1.5 rounded bg-command-elevated hover:bg-command-border text-command-text transition-colors"
              title="Step Back 1s"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              onClick={togglePlay}
              className="px-3 py-1.5 rounded bg-radar-bright hover:bg-radar-green text-command-bg font-bold flex items-center space-x-1.5 transition-all shadow-radar"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isPlaying ? 'Pause' : 'Play'}</span>
            </button>
            <button
              onClick={() => handleStep(1.0)}
              className="p-1.5 rounded bg-command-elevated hover:bg-command-border text-command-text transition-colors"
              title="Step Forward 1s"
            >
              <SkipForward className="w-4 h-4" />
            </button>
            <button
              onClick={cycleSpeed}
              className="px-2 py-1 rounded bg-command-elevated hover:bg-command-border text-command-muted hover:text-command-text border border-command-border"
            >
              {playbackRate}x
            </button>
          </div>

          {/* Right Confidence Filter Slider */}
          <div className="flex items-center space-x-3 bg-command-elevated px-3 py-1 rounded-md border border-command-border">
            <Sliders className="w-3.5 h-3.5 text-radar-bright" />
            <span className="text-command-muted">Threshold:</span>
            <input
              type="range"
              min="0.10"
              max="0.80"
              step="0.05"
              value={confidenceFilter}
              onChange={(e) => onConfidenceFilterChange(parseFloat(e.target.value))}
              className="w-24 accent-radar-bright"
            />
            <span className="text-radar-bright font-bold">{(confidenceFilter * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
