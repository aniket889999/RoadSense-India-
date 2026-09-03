'use client';

import React, { useState } from 'react';
import { X, UploadCloud, Film, CheckCircle2, AlertCircle, Settings2 } from 'lucide-react';
import { uploadSession, triggerProcessing } from '../lib/api';
import { DriveSession } from '../lib/types';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSessionCreated: (session: DriveSession) => void;
}

export function UploadModal({ isOpen, onClose, onSessionCreated }: UploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Advanced inference tuning controls
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [samplingFps, setSamplingFps] = useState(5.0);
  const [maxFrames, setMaxFrames] = useState(150);
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (!isOpen) return null;

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.includes('video') || file.name.match(/\.(mp4|mov|avi|mkv)$/i)) {
        setSelectedFile(file);
        setUploadError(null);
      } else {
        setUploadError('Please select a valid video file (.mp4, .mov, .avi).');
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setUploadError(null);
    }
  };

  const handleStartProcessing = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setUploadError(null);

    try {
      // 1. Upload video to local spool
      const session = await uploadSession(selectedFile);

      // 2. Queue background inference
      await triggerProcessing(session.id, {
        confidence_threshold: confidenceThreshold,
        sampling_fps: samplingFps,
        max_frames: maxFrames,
      });

      onSessionCreated(session);
      onClose();
    } catch (err: any) {
      console.error('Failed to upload and start session:', err);
      setUploadError(err.message || 'Upload failed. Please check local API connection.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-xl glass-panel-elevated p-6 space-y-5 shadow-2xl border border-command-border">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-command-border pb-3">
          <div className="flex items-center space-x-2">
            <Film className="w-5 h-5 text-radar-bright" />
            <h3 className="text-sm font-bold text-command-text tracking-wide">
              Ingest Dashcam Recording
            </h3>
          </div>
          <button
            onClick={onClose}
            disabled={isUploading}
            className="text-command-muted hover:text-command-text p-1 rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Dropzone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleFileDrop}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all ${
            selectedFile
              ? 'border-radar-bright bg-radar-dim/10'
              : 'border-command-border hover:border-command-subtle bg-command-surface/50'
          }`}
          onClick={() => document.getElementById('dashcam-upload-input')?.click()}
        >
          <input
            id="dashcam-upload-input"
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi"
            className="hidden"
            onChange={handleFileSelect}
            disabled={isUploading}
          />
          <div className="flex flex-col items-center space-y-2">
            {selectedFile ? (
              <>
                <CheckCircle2 className="w-10 h-10 text-radar-bright" />
                <span className="text-xs font-semibold text-command-text truncate max-w-xs">
                  {selectedFile.name}
                </span>
                <span className="text-[11px] font-mono text-command-muted">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready to Process
                </span>
              </>
            ) : (
              <>
                <UploadCloud className="w-10 h-10 text-command-muted" />
                <span className="text-xs font-medium text-command-text">
                  Drag & Drop dashcam video or <span className="text-radar-bright">Browse</span>
                </span>
                <span className="text-[11px] font-mono text-command-muted">
                  Supports MP4, MOV, AVI (Local storage only, 0 cloud transmission)
                </span>
              </>
            )}
          </div>
        </div>

        {/* Error notice */}
        {uploadError && (
          <div className="flex items-center space-x-2 p-3 rounded bg-accent-red/10 border border-accent-red/30 text-accent-red text-xs">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{uploadError}</span>
          </div>
        )}

        {/* Advanced tuning expandable */}
        <div className="border-t border-command-border pt-3">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-2 text-xs text-command-muted hover:text-command-text font-mono transition-colors"
          >
            <Settings2 className="w-4 h-4" />
            <span>{showAdvanced ? 'Hide Inference Parameters' : 'Tune Inference Parameters'}</span>
          </button>

          {showAdvanced && (
            <div className="mt-3 p-3 rounded-lg bg-command-surface border border-command-border space-y-3 text-xs font-mono">
              <div className="flex justify-between items-center">
                <span className="text-command-muted">Confidence Threshold:</span>
                <span className="text-radar-bright font-bold">{confidenceThreshold}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.80"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full accent-radar-bright"
              />

              <div className="flex justify-between items-center">
                <span className="text-command-muted">Sampling Cadence:</span>
                <span className="text-accent-cyan font-bold">{samplingFps} FPS</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="15.0"
                step="1.0"
                value={samplingFps}
                onChange={(e) => setSamplingFps(parseFloat(e.target.value))}
                className="w-full accent-accent-cyan"
              />

              <div className="flex justify-between items-center">
                <span className="text-command-muted">Max Sampled Frames:</span>
                <span className="text-accent-amber font-bold">{maxFrames} frames</span>
              </div>
              <input
                type="range"
                min="30"
                max="300"
                step="30"
                value={maxFrames}
                onChange={(e) => setMaxFrames(parseInt(e.target.value))}
                className="w-full accent-accent-amber"
              />
            </div>
          )}
        </div>

        {/* Modal Actions */}
        <div className="flex justify-end space-x-3 pt-2">
          <button
            onClick={onClose}
            disabled={isUploading}
            className="px-4 py-2 rounded-md bg-command-elevated hover:bg-command-border text-command-text text-xs border border-command-border transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleStartProcessing}
            disabled={!selectedFile || isUploading}
            className="px-5 py-2 rounded-md bg-radar-bright hover:bg-radar-green disabled:opacity-50 disabled:cursor-not-allowed text-command-bg font-bold text-xs transition-all shadow-radar"
          >
            {isUploading ? 'Spooling & Launching...' : 'Start Drive Review'}
          </button>
        </div>
      </div>
    </div>
  );
}
