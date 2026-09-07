'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  ApproachZone,
  Camera,
  LayoutRevisionStatus,
  LayoutValidationError,
  NormalizedPoint,
  ParkingLayoutRevision,
  ParkingSpace,
  Site,
  SpaceType,
} from '../lib/types';
import {
  createCamera,
  createDraftLayout,
  createSite,
  fetchCameraLayouts,
  fetchSiteCameras,
  fetchSites,
  getCameraReferenceImageUrl,
  invalidateCameraCalibration,
  submitLayout,
  updateLayout,
  uploadCameraReferenceImage,
  validateLayout,
  verifyLayout,
} from '../lib/parkingApi';
import { DrawingMode, ParkingCanvas } from './ParkingCanvas';
import { LayoutVerificationModal } from './LayoutVerificationModal';
import { LayoutSubmitModal } from './LayoutSubmitModal';
import { ReferenceImageReplaceModal } from './ReferenceImageReplaceModal';
import { CameraCalibrationModal } from './CameraCalibrationModal';
import { CameraStabilityPanel } from './CameraStabilityPanel';
import { ParkingOccupancyPanel } from './ParkingOccupancyPanel';
import {
  AlertCircle,
  AlertTriangle,
  Camera as CameraIcon,
  Car,
  CheckCircle2,
  ChevronRight,
  Clock,
  Edit3,
  Fingerprint,
  Layers,
  MapPin,
  Maximize2,
  MousePointer,
  PenTool,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Send,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  UploadCloud,
  XCircle,
} from 'lucide-react';

export function ParkingLayoutView() {
  // Navigation & Entity State
  const [sites, setSites] = useState<Site[]>([]);
  const [selectedSiteId, setSelectedSiteId] = useState<string | null>(null);

  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);

  const [layouts, setLayouts] = useState<ParkingLayoutRevision[]>([]);
  const [activeLayout, setActiveLayout] = useState<ParkingLayoutRevision | null>(null);

  // Editor Geometry State
  const [parkingSpaces, setParkingSpaces] = useState<ParkingSpace[]>([]);
  const [approachZones, setApproachZones] = useState<ApproachZone[]>([]);
  const [selectedSpaceId, setSelectedSpaceId] = useState<string | null>(null);

  // Drawing & Interaction State
  const [drawingMode, setDrawingMode] = useState<DrawingMode>('idle');
  const [currentPoints, setCurrentPoints] = useState<NormalizedPoint[]>([]);

  // Validation & Audit State
  const [validationErrors, setValidationErrors] = useState<LayoutValidationError[]>([]);
  const [isValidating, setIsValidating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isBranchingDraft, setIsBranchingDraft] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [activeRightTab, setActiveRightTab] = useState<'editor' | 'stability' | 'occupancy'>('editor');

  // Modals
  const [isSubmitModalOpen, setIsSubmitModalOpen] = useState(false);
  const [isVerifyModalOpen, setIsVerifyModalOpen] = useState(false);
  const [isReplaceImageModalOpen, setIsReplaceImageModalOpen] = useState(false);
  const [pendingImageFile, setPendingImageFile] = useState<File | null>(null);
  const [isInvalidateModalOpen, setIsInvalidateModalOpen] = useState(false);
  const [isNewSiteModalOpen, setIsNewSiteModalOpen] = useState(false);
  const [isNewCameraModalOpen, setIsNewCameraModalOpen] = useState(false);

  // New Entity Form State
  const [newSiteName, setNewSiteName] = useState('');
  const [newCameraName, setNewCameraName] = useState('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load Sites
  const loadSites = useCallback(async () => {
    try {
      const siteList = await fetchSites();
      setSites(siteList);
      if (siteList.length > 0 && !selectedSiteId) {
        setSelectedSiteId(siteList[0].id);
      }
    } catch (err: any) {
      console.error('Failed to load sites:', err);
    }
  }, [selectedSiteId]);

  useEffect(() => {
    loadSites();
  }, [loadSites]);

  // Load Cameras when selectedSiteId changes
  useEffect(() => {
    if (!selectedSiteId) {
      setCameras([]);
      setSelectedCameraId(null);
      return;
    }

    fetchSiteCameras(selectedSiteId)
      .then((camList) => {
        setCameras(camList);
        if (camList.length > 0) {
          setSelectedCameraId(camList[0].id);
        } else {
          setSelectedCameraId(null);
          setLayouts([]);
          setActiveLayout(null);
          setParkingSpaces([]);
          setApproachZones([]);
        }
      })
      .catch((err) => console.error('Failed to load cameras:', err));
  }, [selectedSiteId]);

  // Load Layouts when selectedCameraId changes
  // STEP 6.8: Choose the latest editable draft by default when present; otherwise show active verified
  const loadCameraLayouts = useCallback(async (camId: string, preferredRevisionId?: string) => {
    try {
      const revs = await fetchCameraLayouts(camId);
      setLayouts(revs);
      if (revs.length > 0) {
        let primary: ParkingLayoutRevision;
        if (preferredRevisionId) {
          primary = revs.find((r) => r.id === preferredRevisionId) || revs[0];
        } else {
          const latestDraft = revs.find((r) => r.status === 'DRAFT');
          const activeVerified = revs.find((r) => r.status === 'VERIFIED');
          primary = latestDraft || activeVerified || revs[0];
        }

        setActiveLayout(primary);
        setParkingSpaces(primary.parking_spaces || []);
        setApproachZones(primary.approach_zones || []);
        if (primary.parking_spaces?.length > 0) {
          setSelectedSpaceId(primary.parking_spaces[0].id || null);
        } else {
          setSelectedSpaceId(null);
        }
      } else {
        setActiveLayout(null);
        setParkingSpaces([]);
        setApproachZones([]);
        setSelectedSpaceId(null);
      }
      setValidationErrors([]);
    } catch (err: any) {
      console.error('Failed to load layouts:', err);
    }
  }, []);

  useEffect(() => {
    if (selectedCameraId) {
      loadCameraLayouts(selectedCameraId);
    }
  }, [selectedCameraId, loadCameraLayouts]);

  const activeCamera = cameras.find((c) => c.id === selectedCameraId);
  const referenceImageUrl = activeCamera?.reference_image_path
    ? getCameraReferenceImageUrl(activeCamera.id)
    : null;

  // STEP 6.6: Editing is enabled ONLY for DRAFT revisions
  const isDraft = activeLayout ? activeLayout.status === 'DRAFT' : true;

  // Handle image file selection
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !selectedCameraId) return;

    // Check if camera already has any layouts
    const hasExistingLayouts = layouts.length > 0;
    if (hasExistingLayouts) {
      setPendingImageFile(file);
      setIsReplaceImageModalOpen(true);
    } else {
      // First upload: upload directly
      performDirectImageUpload(file);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const performDirectImageUpload = async (file: File) => {
    if (!selectedCameraId) return;
    try {
      const updatedCam = await uploadCameraReferenceImage(selectedCameraId, file);
      setCameras((prev) => prev.map((c) => (c.id === updatedCam.id ? updatedCam : c)));
      setFeedbackMessage({ type: 'success', text: 'Reference image uploaded and calibrated!' });
      await loadCameraLayouts(selectedCameraId);
    } catch (err: any) {
      setFeedbackMessage({ type: 'error', text: err.message || 'Failed to upload reference image' });
    }
  };

  const handleConfirmReplaceImage = async (file: File, operatorLabel: string, reason: string) => {
    if (!selectedCameraId) return;
    const updatedCam = await uploadCameraReferenceImage(selectedCameraId, file, true, operatorLabel, reason);
    setCameras((prev) => prev.map((c) => (c.id === updatedCam.id ? updatedCam : c)));
    setFeedbackMessage({
      type: 'success',
      text: 'Reference image replaced! All previous layouts have been invalidated for occupancy safety.',
    });
    await loadCameraLayouts(selectedCameraId);
  };

  // Drawing Handlers
  const handleAddPoint = (point: NormalizedPoint) => {
    if (!isDraft) return;
    setCurrentPoints((prev) => [...prev, point]);
  };

  const handleClosePolygon = () => {
    if (!isDraft) return;
    if (currentPoints.length < 3) {
      setFeedbackMessage({ type: 'error', text: 'Polygons require at least 3 vertices.' });
      return;
    }

    if (drawingMode === 'draw_space') {
      const nextIndex = parkingSpaces.length + 1;
      const newSpace: ParkingSpace = {
        id: `space_temp_${Date.now()}`,
        operator_label: `Bay-${nextIndex < 10 ? '0' + nextIndex : nextIndex}`,
        space_type: 'STANDARD',
        polygon_normalized: currentPoints,
        active: true,
      };
      setParkingSpaces((prev) => [...prev, newSpace]);
      setSelectedSpaceId(newSpace.id!);
      setFeedbackMessage({ type: 'success', text: `Added parking space ${newSpace.operator_label}.` });
    } else if (drawingMode === 'draw_approach' && selectedSpaceId) {
      const newApproach: ApproachZone = {
        id: `approach_temp_${Date.now()}`,
        parking_space_id: selectedSpaceId,
        polygon_normalized: currentPoints,
      };
      setApproachZones((prev) => [...prev.filter((a) => a.parking_space_id !== selectedSpaceId), newApproach]);
      setFeedbackMessage({ type: 'success', text: 'Attached approach zone to selected space.' });
    }

    setCurrentPoints([]);
    setDrawingMode('idle');
  };

  const handleUpdateSpacePolygon = (spaceId: string, newPoints: NormalizedPoint[]) => {
    if (!isDraft) return;
    setParkingSpaces((prev) =>
      prev.map((s) => (s.id === spaceId ? { ...s, polygon_normalized: newPoints } : s))
    );
  };

  const handleUpdateApproachPolygon = (spaceId: string, newPoints: NormalizedPoint[]) => {
    if (!isDraft) return;
    setApproachZones((prev) =>
      prev.map((a) => (a.parking_space_id === spaceId ? { ...a, polygon_normalized: newPoints } : a))
    );
  };

  const handleDeleteSelectedSpace = () => {
    if (!isDraft || !selectedSpaceId) return;
    setParkingSpaces((prev) => prev.filter((s) => s.id !== selectedSpaceId));
    setApproachZones((prev) => prev.filter((a) => a.parking_space_id !== selectedSpaceId));
    setSelectedSpaceId(null);
  };

  const handleClearDraft = () => {
    if (!isDraft) return;
    if (window.confirm('Are you sure you want to clear all drawn polygons in this draft?')) {
      setParkingSpaces([]);
      setApproachZones([]);
      setSelectedSpaceId(null);
      setCurrentPoints([]);
      setDrawingMode('idle');
    }
  };

  // STEP 6.1 & 6.2: One-click Save and Validate
  const handleSaveAndValidate = async () => {
    if (!selectedCameraId) return;
    setIsValidating(true);
    try {
      let saved: ParkingLayoutRevision;
      if (!activeLayout) {
        saved = await createDraftLayout(selectedCameraId, parkingSpaces, approachZones);
      } else {
        saved = await updateLayout(activeLayout.id, parkingSpaces, approachZones);
      }

      // Validate the EXACT returned layout ID
      const valResult = await validateLayout(saved.id);
      setValidationErrors(valResult.errors);

      // Refresh layout list and update active layout with returned object
      const refreshedRevs = await fetchCameraLayouts(selectedCameraId);
      setLayouts(refreshedRevs);
      const updatedActive = refreshedRevs.find((r) => r.id === saved.id) || saved;
      setActiveLayout(updatedActive);
      setParkingSpaces(updatedActive.parking_spaces || []);
      setApproachZones(updatedActive.approach_zones || []);

      if (valResult.is_valid) {
        setFeedbackMessage({
          type: 'success',
          text: `Layout Rev #${saved.revision_number} saved and validated! Zero geometry errors.`,
        });
      } else {
        setFeedbackMessage({
          type: 'error',
          text: `Validation found ${valResult.errors.length} errors. Review issues below.`,
        });
      }
    } catch (err: any) {
      setFeedbackMessage({ type: 'error', text: err.message || 'Save and validate failed' });
    } finally {
      setIsValidating(false);
    }
  };

  // STEP 6.6: Explicitly branch a new draft from any non-draft revision
  const handleCreateNewDraftFromRevision = async () => {
    if (!activeLayout || !selectedCameraId) return;
    setIsBranchingDraft(true);
    try {
      const newDraft = await updateLayout(activeLayout.id, parkingSpaces, approachZones);
      await loadCameraLayouts(selectedCameraId, newDraft.id);
      setFeedbackMessage({
        type: 'success',
        text: `Created new editable draft Rev #${newDraft.revision_number} branched from Rev #${activeLayout.revision_number}!`,
      });
    } catch (err: any) {
      setFeedbackMessage({ type: 'error', text: err.message || 'Failed to branch new draft' });
    } finally {
      setIsBranchingDraft(false);
    }
  };

  const handleSaveDraftOnly = async () => {
    if (!selectedCameraId) return;
    setIsSaving(true);
    try {
      let saved: ParkingLayoutRevision;
      if (!activeLayout) {
        saved = await createDraftLayout(selectedCameraId, parkingSpaces, approachZones);
      } else {
        saved = await updateLayout(activeLayout.id, parkingSpaces, approachZones);
      }
      setFeedbackMessage({ type: 'success', text: `Layout draft saved (Rev #${saved.revision_number}).` });
      await loadCameraLayouts(selectedCameraId, saved.id);
    } catch (err: any) {
      setFeedbackMessage({ type: 'error', text: err.message || 'Failed to save draft' });
    } finally {
      setIsSaving(false);
    }
  };

  // STEP 6.3: Submission modal with entered operator label
  const handleSubmitConfirm = async (operatorLabel: string, note?: string) => {
    if (!activeLayout || !selectedCameraId) return;
    const res = await submitLayout(activeLayout.id, operatorLabel, note);
    setFeedbackMessage({ type: 'success', text: `Layout submitted for human review (Rev #${res.revision_number}).` });
    await loadCameraLayouts(selectedCameraId, res.id);
  };

  const handleVerifyConfirm = async (operatorLabel: string, note?: string) => {
    if (!activeLayout || !selectedCameraId) return;
    const res = await verifyLayout(activeLayout.id, operatorLabel, true, note);
    setFeedbackMessage({ type: 'success', text: `Layout Rev #${res.revision_number} successfully verified and activated!` });
    await loadCameraLayouts(selectedCameraId, res.id);
    if (selectedSiteId) {
      const camList = await fetchSiteCameras(selectedSiteId);
      setCameras(camList);
    }
  };

  const handleInvalidateConfirm = async (operatorLabel: string, reason: string, note?: string) => {
    if (!selectedCameraId) return;
    const res = await invalidateCameraCalibration(selectedCameraId, operatorLabel, reason, note);
    setCameras((prev) => prev.map((c) => (c.id === res.id ? res : c)));
    setFeedbackMessage({ type: 'success', text: `Camera calibration invalidated.` });
    await loadCameraLayouts(selectedCameraId);
  };

  const handleCreateSite = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSiteName.trim()) return;
    try {
      const site = await createSite(newSiteName.trim());
      setSites((prev) => [site, ...prev]);
      setSelectedSiteId(site.id);
      setIsNewSiteModalOpen(false);
      setNewSiteName('');
      setFeedbackMessage({ type: 'success', text: `Created site "${site.name}".` });
    } catch (err: any) {
      setFeedbackMessage({ type: 'error', text: err.message || 'Failed to create site' });
    }
  };

  const handleCreateCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedSiteId || !newCameraName.trim()) return;
    try {
      const cam = await createCamera(selectedSiteId, newCameraName.trim());
      setCameras((prev) => [cam, ...prev]);
      setSelectedCameraId(cam.id);
      setIsNewCameraModalOpen(false);
      setNewCameraName('');
      setFeedbackMessage({ type: 'success', text: `Created camera "${cam.name}".` });
    } catch (err: any) {
      setFeedbackMessage({ type: 'error', text: err.message || 'Failed to create camera' });
    }
  };

  const selectedSpace = parkingSpaces.find((s) => s.id === selectedSpaceId) || null;
  const selectedApproach = approachZones.find((a) => a.parking_space_id === selectedSpaceId) || null;

  return (
    <div className="flex flex-col h-[calc(100vh-8.5rem)] space-y-3 font-mono">
      {/* Disclaimer Banner */}
      <div className="flex items-center justify-between px-4 py-2 rounded-lg bg-command-elevated border border-command-border text-xs">
        <div className="flex items-center space-x-2 text-radar-bright font-bold">
          <Layers className="w-4 h-4" />
          <span>RoadSense SiteOps: Parking Geometry & ROI Configuration</span>
        </div>
        <div className="flex items-center space-x-3 text-command-muted text-[11px]">
          <span className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-accent-amber animate-pulse" />
            <span>Layout configuration only — occupancy inference is not connected.</span>
          </span>
          <button
            id="stability-gate-shortcut-btn"
            onClick={() => setActiveRightTab('stability')}
            className="flex items-center space-x-1 px-2 py-0.5 rounded bg-rose-950/60 border border-rose-600/60 text-rose-300 hover:text-white font-bold transition-all"
            title="Inspect Fail-Closed Stability Gate"
          >
            <ShieldAlert className="w-3 h-3 text-rose-400" />
            <span>Stability Gate</span>
          </button>
        </div>
      </div>

      {/* Top Controls Bar */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 p-3 rounded-xl glass-panel border border-command-border text-xs">
        {/* Site Selector */}
        <div className="lg:col-span-3 flex items-center space-x-2">
          <MapPin className="w-4 h-4 text-command-muted shrink-0" />
          <div className="flex-1">
            <select
              value={selectedSiteId || ''}
              onChange={(e) => setSelectedSiteId(e.target.value)}
              className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
            >
              {sites.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.cameras_count} Cams)
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() => setIsNewSiteModalOpen(true)}
            className="p-1.5 rounded bg-command-elevated border border-command-border hover:text-radar-bright"
            title="Create New Site"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {/* Camera Selector */}
        <div className="lg:col-span-3 flex items-center space-x-2">
          <CameraIcon className="w-4 h-4 text-command-muted shrink-0" />
          <div className="flex-1">
            <select
              value={selectedCameraId || ''}
              onChange={(e) => setSelectedCameraId(e.target.value)}
              disabled={cameras.length === 0}
              className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright disabled:opacity-50"
            >
              {cameras.length === 0 ? (
                <option value="">No cameras configured</option>
              ) : (
                cameras.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name} [{c.calibration_status}]
                  </option>
                ))
              )}
            </select>
          </div>
          <button
            onClick={() => setIsNewCameraModalOpen(true)}
            disabled={!selectedSiteId}
            className="p-1.5 rounded bg-command-elevated border border-command-border hover:text-radar-bright disabled:opacity-50"
            title="Create New Camera"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {/* Upload Reference Image */}
        <div className="lg:col-span-3 flex items-center space-x-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInputChange}
            accept="image/jpeg,image/png"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={!selectedCameraId}
            className="w-full flex items-center justify-center space-x-2 px-3 py-1.5 rounded bg-command-elevated border border-command-border hover:border-radar-bright text-command-text hover:text-radar-bright transition-all disabled:opacity-50"
          >
            <UploadCloud className="w-4 h-4" />
            <span>{activeCamera?.reference_image_path ? 'Replace Reference Frame' : 'Upload Reference Frame'}</span>
          </button>
        </div>

        {/* Layout Revisions Selector */}
        <div className="lg:col-span-3 flex items-center space-x-2">
          <Layers className="w-4 h-4 text-command-muted shrink-0" />
          <div className="flex-1">
            <select
              value={activeLayout?.id || ''}
              onChange={(e) => {
                const rev = layouts.find((l) => l.id === e.target.value);
                if (rev) {
                  setActiveLayout(rev);
                  setParkingSpaces(rev.parking_spaces || []);
                  setApproachZones(rev.approach_zones || []);
                  setValidationErrors([]);
                }
              }}
              disabled={layouts.length === 0}
              className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright disabled:opacity-50"
            >
              {layouts.length === 0 ? (
                <option value="">No layout revisions</option>
              ) : (
                layouts.map((l) => (
                  <option key={l.id} value={l.id}>
                    Rev #{l.revision_number} [{l.status}]
                  </option>
                ))
              )}
            </select>
          </div>
        </div>
      </div>

      {/* Main Workspace Stage */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 flex-1 overflow-hidden">
        {/* Canvas & Drawing Tools (8 Cols) */}
        <div className="lg:col-span-8 flex flex-col h-full space-y-2">
          {/* Drawing Toolbar */}
          <div className="flex flex-wrap items-center justify-between p-2 rounded-lg bg-command-surface border border-command-border text-xs gap-2">
            <div className="flex items-center space-x-1.5">
              {isDraft ? (
                <>
                  <button
                    onClick={() => {
                      setDrawingMode('draw_space');
                      setCurrentPoints([]);
                    }}
                    className={`flex items-center space-x-1.5 px-3 py-1.5 rounded transition-all ${
                      drawingMode === 'draw_space'
                        ? 'bg-blue-600 text-white shadow-sm'
                        : 'bg-command-elevated text-command-text hover:bg-command-elevated/70'
                    }`}
                  >
                    <PenTool className="w-3.5 h-3.5" />
                    <span>Draw Space</span>
                  </button>

                  <button
                    onClick={() => {
                      if (!selectedSpaceId) {
                        setFeedbackMessage({ type: 'info', text: 'Select a space first to attach an approach zone.' });
                        return;
                      }
                      setDrawingMode('draw_approach');
                      setCurrentPoints([]);
                    }}
                    disabled={!selectedSpaceId}
                    className={`flex items-center space-x-1.5 px-3 py-1.5 rounded transition-all ${
                      drawingMode === 'draw_approach'
                        ? 'bg-purple-600 text-white shadow-sm'
                        : 'bg-command-elevated text-command-text hover:bg-command-elevated/70 disabled:opacity-50'
                    }`}
                  >
                    <Maximize2 className="w-3.5 h-3.5" />
                    <span>Draw Approach</span>
                  </button>

                  <button
                    onClick={() => {
                      setDrawingMode('edit_vertices');
                      setCurrentPoints([]);
                    }}
                    className={`flex items-center space-x-1.5 px-3 py-1.5 rounded transition-all ${
                      drawingMode === 'edit_vertices'
                        ? 'bg-cyan-600 text-white shadow-sm'
                        : 'bg-command-elevated text-command-text hover:bg-command-elevated/70'
                    }`}
                  >
                    <MousePointer className="w-3.5 h-3.5" />
                    <span>Edit Vertices</span>
                  </button>

                  {currentPoints.length > 0 && (
                    <>
                      <button
                        onClick={() => setCurrentPoints((prev) => prev.slice(0, -1))}
                        className="flex items-center space-x-1 px-2.5 py-1.5 rounded bg-command-elevated text-command-muted hover:text-command-text"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        <span>Undo Point</span>
                      </button>
                      <button
                        onClick={handleClosePolygon}
                        disabled={currentPoints.length < 3}
                        className="flex items-center space-x-1 px-2.5 py-1.5 rounded bg-emerald-600 text-white font-bold disabled:opacity-50"
                      >
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Close Polygon</span>
                      </button>
                    </>
                  )}
                </>
              ) : (
                <div className="flex items-center space-x-2 text-command-muted text-xs">
                  <span className="px-2 py-1 rounded bg-command-elevated border border-command-border">
                    Revision #{activeLayout?.revision_number} is {activeLayout?.status} (Read-only)
                  </span>
                  <button
                    onClick={handleCreateNewDraftFromRevision}
                    disabled={isBranchingDraft}
                    className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white font-bold shadow-sm transition-all disabled:opacity-50"
                  >
                    <Edit3 className="w-3.5 h-3.5" />
                    <span>{isBranchingDraft ? 'Branching...' : 'Create New Draft From This Revision'}</span>
                  </button>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-1.5">
              {isDraft && (
                <>
                  <button
                    onClick={handleClearDraft}
                    className="px-2.5 py-1.5 rounded bg-command-elevated text-accent-red hover:bg-accent-red/20"
                    title="Clear All Draft Polygons"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={handleSaveDraftOnly}
                    disabled={isSaving}
                    className="flex items-center space-x-1 px-3 py-1.5 rounded bg-command-elevated border border-command-border text-command-text hover:border-radar-bright font-bold disabled:opacity-50"
                  >
                    <Save className="w-3.5 h-3.5" />
                    <span>{isSaving ? 'Saving...' : 'Save Draft'}</span>
                  </button>
                </>
              )}

              {/* STEP 6.1: One-Click Save and Validate */}
              <button
                onClick={handleSaveAndValidate}
                disabled={isValidating || isSaving}
                className="flex items-center space-x-1 px-3 py-1.5 rounded bg-radar-bright hover:bg-radar-green text-command-bg font-bold shadow-radar transition-all disabled:opacity-50"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>{isValidating ? 'Validating...' : 'Save & Validate'}</span>
              </button>
            </div>
          </div>

          {/* Feedback Banner */}
          {feedbackMessage && (
            <div
              className={`px-3 py-2 rounded-lg text-xs flex items-center justify-between ${
                feedbackMessage.type === 'success'
                  ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
                  : feedbackMessage.type === 'error'
                  ? 'bg-red-500/10 border border-red-500/30 text-red-400'
                  : 'bg-blue-500/10 border border-blue-500/30 text-blue-400'
              }`}
            >
              <span>{feedbackMessage.text}</span>
              <button onClick={() => setFeedbackMessage(null)} className="text-command-muted hover:text-command-text">
                &times;
              </button>
            </div>
          )}

          {/* Interactive Canvas */}
          <div className="flex-1 min-h-0">
            <ParkingCanvas
              referenceImageUrl={referenceImageUrl}
              status={activeLayout?.status || 'DRAFT'}
              calibrationStatus={activeCamera?.calibration_status || 'NOT_CONFIGURED'}
              parkingSpaces={parkingSpaces}
              approachZones={approachZones}
              selectedSpaceId={selectedSpaceId}
              drawingMode={drawingMode}
              currentPoints={currentPoints}
              validationErrors={validationErrors}
              onSelectSpace={setSelectedSpaceId}
              onAddPoint={handleAddPoint}
              onClosePolygon={handleClosePolygon}
              onUpdateSpacePolygon={handleUpdateSpacePolygon}
              onUpdateApproachPolygon={handleUpdateApproachPolygon}
            />
          </div>
        </div>

        {/* Right Inspector & Audit Panel (4 Cols) */}
        <div className="lg:col-span-4 flex flex-col h-full space-y-3 overflow-y-auto pr-1">
          {/* Sub-tab Navigation */}
          <div className="flex rounded-lg bg-command-elevated border border-command-border p-1 text-xs">
            <button
              onClick={() => setActiveRightTab('editor')}
              className={`flex-1 flex items-center justify-center space-x-1.5 py-1.5 rounded transition-all font-bold ${
                activeRightTab === 'editor'
                  ? 'bg-command-surface text-radar-bright shadow-sm border border-command-border'
                  : 'text-command-muted hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>ROI & Governance</span>
            </button>
            <button
              id="camera-stability-tab-button"
              onClick={() => setActiveRightTab('stability')}
              className={`flex-1 flex items-center justify-center space-x-1.5 py-1.5 rounded transition-all font-bold ${
                activeRightTab === 'stability'
                  ? 'bg-command-surface text-amber-400 shadow-sm border border-command-border'
                  : 'text-command-muted hover:text-white'
              }`}
            >
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Camera Stability Gate</span>
            </button>
            <button
              id="parking-occupancy-tab-button"
              onClick={() => setActiveRightTab('occupancy')}
              className={`flex-1 flex items-center justify-center space-x-1.5 py-1.5 rounded transition-all font-bold ${
                activeRightTab === 'occupancy'
                  ? 'bg-command-surface text-cyan-400 shadow-sm border border-command-border'
                  : 'text-command-muted hover:text-white'
              }`}
            >
              <Car className="w-3.5 h-3.5" />
              <span>Occupancy Pipeline</span>
            </button>
          </div>

          {activeRightTab === 'occupancy' ? (
            <ParkingOccupancyPanel
              cameraId={selectedCameraId}
              cameraName={activeCamera?.name}
              activeLayoutId={activeLayout?.id}
              onNavigateToStability={() => setActiveRightTab('stability')}
              onNavigateToLayout={() => setActiveRightTab('editor')}
            />
          ) : activeRightTab === 'stability' ? (
            <CameraStabilityPanel
              cameraId={selectedCameraId}
              cameraName={activeCamera?.name}
              hasReferenceImage={Boolean(activeCamera?.reference_image_path)}
              activeLayoutId={activeLayout?.id}
              onCalibrationInvalidated={async () => {
                if (selectedSiteId) {
                  const camList = await fetchSiteCameras(selectedSiteId);
                  setCameras(camList);
                }
                if (selectedCameraId) {
                  await loadCameraLayouts(selectedCameraId);
                }
              }}
            />
          ) : (
            <>
              {/* Selected Space Properties Panel */}
              <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3 text-xs">
            <div className="flex items-center justify-between border-b border-command-border pb-2">
              <span className="font-bold text-command-text uppercase tracking-wider">
                {selectedSpace ? `Inspecting ${selectedSpace.operator_label}` : 'Space Inspector'}
              </span>
              {selectedSpace && isDraft && (
                <button
                  onClick={handleDeleteSelectedSpace}
                  className="text-accent-red hover:underline flex items-center space-x-1"
                >
                  <Trash2 className="w-3 h-3" />
                  <span>Delete</span>
                </button>
              )}
            </div>

            {selectedSpace ? (
              <div className="space-y-2.5">
                <div>
                  <label className="text-[10px] text-command-muted uppercase">Operator Label</label>
                  <input
                    type="text"
                    value={selectedSpace.operator_label}
                    disabled={!isDraft}
                    onChange={(e) => {
                      const newLabel = e.target.value;
                      setParkingSpaces((prev) =>
                        prev.map((s) => (s.id === selectedSpace.id ? { ...s, operator_label: newLabel } : s))
                      );
                    }}
                    className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright disabled:opacity-50"
                  />
                </div>

                <div>
                  <label className="text-[10px] text-command-muted uppercase">Space Type</label>
                  <select
                    value={selectedSpace.space_type}
                    disabled={!isDraft}
                    onChange={(e) => {
                      const newType = e.target.value as SpaceType;
                      setParkingSpaces((prev) =>
                        prev.map((s) => (s.id === selectedSpace.id ? { ...s, space_type: newType } : s))
                      );
                    }}
                    className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright disabled:opacity-50"
                  >
                    <option value="STANDARD">STANDARD</option>
                    <option value="ACCESSIBLE">ACCESSIBLE</option>
                    <option value="EV_CHARGING">EV_CHARGING</option>
                    <option value="LOADING">LOADING</option>
                    <option value="EMERGENCY">EMERGENCY</option>
                    <option value="OTHER">OTHER</option>
                  </select>
                </div>

                <div className="p-2.5 rounded bg-command-elevated/70 border border-command-border space-y-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-command-muted">Approach Zone:</span>
                    <span className={selectedApproach ? 'text-purple-400 font-bold' : 'text-command-muted'}>
                      {selectedApproach ? 'Attached' : 'None'}
                    </span>
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-command-muted">Vertices Count:</span>
                    <span className="text-command-text font-bold">{selectedSpace.polygon_normalized.length}</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-[11px] text-command-muted">
                Click a parking space polygon on the canvas to inspect coordinates, change labels, or attach approach zones.
              </p>
            )}
          </div>

          {/* Validation & Verification Workflow Panel */}
          <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3 text-xs">
            <span className="font-bold text-command-text uppercase tracking-wider block border-b border-command-border pb-2">
              Lifecycle & Governance
            </span>

            <div className="flex items-center justify-between text-[11px]">
              <span className="text-command-muted">Revision Status:</span>
              <span className="font-bold text-command-text">{activeLayout?.status || 'NO LAYOUT'}</span>
            </div>

            {activeLayout?.reference_image_sha256 && (
              <div className="text-[10px] text-command-muted space-y-0.5">
                <span className="text-command-muted uppercase tracking-wider">Snapshot Frame SHA-256:</span>
                <p className="font-mono break-all text-command-text text-[9px]">{activeLayout.reference_image_sha256}</p>
              </div>
            )}

            {activeLayout?.canonical_sha256 && (
              <div className="p-2 rounded bg-command-bg border border-command-border text-[10px] space-y-1">
                <div className="flex items-center space-x-1 text-radar-bright font-bold">
                  <Fingerprint className="w-3.5 h-3.5" />
                  <span>Canonical Fingerprint (SHA-256):</span>
                </div>
                <p className="font-mono break-all text-command-muted">{activeLayout.canonical_sha256}</p>
              </div>
            )}

            {/* Lifecycle Action Buttons */}
            <div className="space-y-2 pt-1">
              {activeLayout?.status === 'DRAFT' && (
                <button
                  onClick={() => setIsSubmitModalOpen(true)}
                  className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded bg-amber-600 hover:bg-amber-500 text-white font-bold shadow-sm"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>Submit for Review</span>
                </button>
              )}

              {activeLayout?.status === 'PENDING_REVIEW' && (
                <button
                  onClick={() => setIsVerifyModalOpen(true)}
                  className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded bg-radar-bright hover:bg-radar-green text-command-bg font-bold shadow-radar"
                >
                  <ShieldCheck className="w-4 h-4" />
                  <span>Verify Layout</span>
                </button>
              )}

              {activeCamera && activeCamera.calibration_status === 'VERIFIED' && (
                <button
                  onClick={() => setIsInvalidateModalOpen(true)}
                  className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded bg-accent-red/20 border border-accent-red/40 hover:bg-accent-red/30 text-accent-red font-bold"
                >
                  <XCircle className="w-3.5 h-3.5" />
                  <span>Invalidate Calibration</span>
                </button>
              )}
            </div>
          </div>

          {/* Validation Errors Panel (if any) */}
          {validationErrors.length > 0 && (
            <div className="p-3.5 rounded-xl bg-accent-red/10 border border-accent-red/30 space-y-2 text-xs">
              <div className="flex items-center space-x-1.5 text-accent-red font-bold">
                <AlertCircle className="w-4 h-4" />
                <span>Validation Errors ({validationErrors.length})</span>
              </div>
              <ul className="space-y-1.5 text-[11px] text-accent-red/90 max-h-40 overflow-y-auto">
                {validationErrors.map((err, idx) => (
                  <li key={idx} className="flex items-start space-x-1.5">
                    <span className="font-bold shrink-0">•</span>
                    <span>
                      {err.space_label ? `[${err.space_label}] ` : ''}
                      {err.message}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Audit History Timeline */}
          <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3 text-xs flex-1 min-h-[160px]">
            <div className="flex items-center space-x-2 text-command-text font-bold uppercase tracking-wider border-b border-command-border pb-2">
              <Clock className="w-4 h-4 text-command-muted" />
              <span>Audit Ledger ({activeLayout?.audit_events?.length || 0})</span>
            </div>

            <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
              {!activeLayout?.audit_events || activeLayout.audit_events.length === 0 ? (
                <p className="text-[11px] text-command-muted italic">No audit events recorded yet.</p>
              ) : (
                activeLayout.audit_events.map((ev) => (
                  <div key={ev.id} className="p-2 rounded bg-command-bg border border-command-border text-[11px] space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-radar-bright">{ev.event_type}</span>
                      <span className="text-[10px] text-command-muted">
                        {new Date(ev.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                    {ev.prior_status && (
                      <div className="text-[10px] text-command-muted">
                        Transition: {ev.prior_status} &rarr; {ev.new_status}
                      </div>
                    )}
                    {ev.local_operator_label && (
                      <div className="text-[10px] text-command-muted">
                        Operator: <span className="text-white font-mono">{ev.local_operator_label}</span>
                      </div>
                    )}
                    {ev.note && <div className="text-[10px] text-command-muted italic">{ev.note}</div>}
                  </div>
                ))
              )}
            </div>
          </div>
            </>
          )}
        </div>
      </div>

      {/* Submission Modal */}
      <LayoutSubmitModal
        isOpen={isSubmitModalOpen}
        onClose={() => setIsSubmitModalOpen(false)}
        onSubmitConfirm={handleSubmitConfirm}
        revisionNumber={activeLayout?.revision_number || 1}
      />

      {/* Verification Modal */}
      <LayoutVerificationModal
        isOpen={isVerifyModalOpen}
        onClose={() => setIsVerifyModalOpen(false)}
        onVerify={handleVerifyConfirm}
        revisionNumber={activeLayout?.revision_number || 1}
        spacesCount={parkingSpaces.length}
      />

      {/* Reference Image Replace Confirmation Modal */}
      <ReferenceImageReplaceModal
        isOpen={isReplaceImageModalOpen}
        onClose={() => {
          setIsReplaceImageModalOpen(false);
          setPendingImageFile(null);
        }}
        onConfirmReplace={handleConfirmReplaceImage}
        pendingFile={pendingImageFile}
        cameraName={activeCamera?.name || 'Camera'}
      />

      {/* Calibration Invalidation Modal */}
      <CameraCalibrationModal
        isOpen={isInvalidateModalOpen}
        onClose={() => setIsInvalidateModalOpen(false)}
        onInvalidate={handleInvalidateConfirm}
        cameraName={activeCamera?.name || 'Camera'}
      />

      {/* New Site Modal */}
      {isNewSiteModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
          <div className="w-full max-w-sm rounded-xl glass-panel-elevated border border-command-border p-5 space-y-4">
            <h3 className="font-bold text-sm text-command-text">Create New Site</h3>
            <form onSubmit={handleCreateSite} className="space-y-3 text-xs">
              <div>
                <label className="block text-command-muted mb-1">Site Name</label>
                <input
                  type="text"
                  value={newSiteName}
                  onChange={(e) => setNewSiteName(e.target.value)}
                  placeholder="e.g. South Terminal Deck"
                  required
                  className="w-full px-3 py-2 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
                />
              </div>
              <div className="flex justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsNewSiteModalOpen(false)}
                  className="px-3 py-1.5 rounded bg-command-elevated text-command-muted"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!newSiteName.trim()}
                  className="px-3 py-1.5 rounded bg-radar-bright text-command-bg font-bold disabled:opacity-50"
                >
                  Create
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* New Camera Modal */}
      {isNewCameraModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
          <div className="w-full max-w-sm rounded-xl glass-panel-elevated border border-command-border p-5 space-y-4">
            <h3 className="font-bold text-sm text-command-text">Create New Camera</h3>
            <form onSubmit={handleCreateCamera} className="space-y-3 text-xs">
              <div>
                <label className="block text-command-muted mb-1">Camera Name</label>
                <input
                  type="text"
                  value={newCameraName}
                  onChange={(e) => setNewCameraName(e.target.value)}
                  placeholder="e.g. North Mast Cam 02"
                  required
                  className="w-full px-3 py-2 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
                />
              </div>
              <div className="flex justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsNewCameraModalOpen(false)}
                  className="px-3 py-1.5 rounded bg-command-elevated text-command-muted"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!newCameraName.trim()}
                  className="px-3 py-1.5 rounded bg-radar-bright text-command-bg font-bold disabled:opacity-50"
                >
                  Create
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
