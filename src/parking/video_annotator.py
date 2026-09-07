"""OpenCV frame annotation and safe FFmpeg stream encoding for parking occupancy video."""

from __future__ import annotations

import collections
import json
import logging
from pathlib import Path
import subprocess
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.parking.occupancy_config import RenderConfig
from src.parking.occupancy_contracts import (
    BayStateSummary,
    FrameOccupancyResult,
    OccupancyState,
    VehicleDetection,
)

logger = logging.getLogger(__name__)


class ParkingVideoAnnotator:
    """
    Renders human-verified parking space polygons and vehicle detections onto frames.
    Visual styling:
      - VACANT: neon-green polygon with independent translucent green fill
      - OCCUPIED: red polygon with independent translucent red fill
      - UNKNOWN: gray polygon with independent translucent gray fill
      - OCCLUDED: amber polygon with independent translucent amber fill
      - Vehicle detection: cyan bounding box with class/track label
      - Bay label: Bay-01 · OCCUPIED · 0.87
      - Summary overlay HUD: Total N | Occupied X | Vacant Y | Unknown Z
    """

    def __init__(self, render_config: RenderConfig) -> None:
        self.config = render_config

    def render_frame(
        self,
        frame_bgr: np.ndarray,
        bay_states: Dict[str, BayStateSummary],
        bay_polygons_px: Dict[str, np.ndarray],
        vehicle_detections: List[VehicleDetection],
        frame_idx: int,
        timestamp_sec: float,
    ) -> np.ndarray:
        """Render complete overlays onto a copy of the input frame."""
        h, w = frame_bgr.shape[:2]
        output = frame_bgr.copy()

        # 1. Render parking space polygons with exact per-state alpha blending
        for bay_id, summary in bay_states.items():
            poly_px = bay_polygons_px.get(bay_id)
            if poly_px is None or len(poly_px) < 3:
                continue

            state = summary.current_state
            if state == OccupancyState.VACANT:
                color = tuple(self.config.vacant_color_bgr)
                alpha = float(self.config.vacant_fill_alpha)
            elif state == OccupancyState.OCCUPIED:
                color = tuple(self.config.occupied_color_bgr)
                alpha = float(self.config.occupied_fill_alpha)
            elif state == OccupancyState.OCCLUDED:
                color = tuple(self.config.occluded_color_bgr)
                alpha = float(self.config.occluded_fill_alpha)
            else:
                color = tuple(self.config.unknown_color_bgr)
                alpha = float(self.config.unknown_fill_alpha)

            min_x = max(0, int(np.min(poly_px[:, 0])))
            max_x = min(w, int(np.max(poly_px[:, 0])) + 1)
            min_y = max(0, int(np.min(poly_px[:, 1])))
            max_y = min(h, int(np.max(poly_px[:, 1])) + 1)

            if max_x > min_x and max_y > min_y:
                roi_h = max_y - min_y
                roi_w = max_x - min_x
                roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                rel_pts = poly_px - np.array([min_x, min_y])
                cv2.fillPoly(roi_mask, [rel_pts], 1)

                roi_orig = output[min_y:max_y, min_x:max_x].astype(np.float32)
                color_arr = np.array(color, dtype=np.float32)
                blended = (1.0 - alpha) * roi_orig + alpha * color_arr
                mask_idx = (roi_mask == 1)
                roi_orig[mask_idx] = blended[mask_idx]
                output[min_y:max_y, min_x:max_x] = np.clip(roi_orig, 0, 255).astype(np.uint8)

        # 2. Draw crisp polygon outlines and bay labels
        for bay_id, summary in bay_states.items():
            poly_px = bay_polygons_px.get(bay_id)
            if poly_px is None or len(poly_px) < 3:
                continue

            state = summary.current_state
            if state == OccupancyState.VACANT:
                color = tuple(self.config.vacant_color_bgr)
            elif state == OccupancyState.OCCUPIED:
                color = tuple(self.config.occupied_color_bgr)
            elif state == OccupancyState.OCCLUDED:
                color = tuple(self.config.occluded_color_bgr)
            else:
                color = tuple(self.config.unknown_color_bgr)

            # Draw polygon border
            cv2.polylines(output, [poly_px], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            if self.config.draw_bay_labels:
                min_x = int(np.min(poly_px[:, 0]))
                min_y = int(np.min(poly_px[:, 1]))
                label_text = f"{summary.operator_label} · {state.value}"
                if summary.confidence > 0.0 and state in (OccupancyState.OCCUPIED, OccupancyState.OCCLUDED):
                    label_text += f" · {summary.confidence:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness = 1
                (lw, lh), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                # Label pill background
                pill_x1 = max(0, min_x)
                pill_y1 = max(0, min_y - lh - 8)
                pill_x2 = min(w - 1, pill_x1 + lw + 8)
                pill_y2 = max(lh + 4, min_y)

                cv2.rectangle(output, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 20, 20), -1)
                cv2.rectangle(output, (pill_x1, pill_y1), (pill_x2, pill_y2), color, 1)
                cv2.putText(
                    output,
                    label_text,
                    (pill_x1 + 4, pill_y2 - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        # 3. Render Vehicle Detections in Cyan
        if self.config.draw_vehicle_boxes:
            box_color = tuple(self.config.vehicle_box_bgr)  # cyan (255, 255, 0)
            for det in vehicle_detections:
                x1, y1, x2, y2 = det.bbox_xyxy
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

                # Draw vehicle box
                cv2.rectangle(output, (ix1, iy1), (ix2, iy2), box_color, 2)

                # Track and Class Label
                if self.config.draw_track_ids and det.track_id is not None:
                    v_label = f"Vehicle #{det.track_id} · {det.class_name} · {det.confidence:.2f}"
                else:
                    v_label = f"{det.class_name} · {det.confidence:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.40
                thickness = 1
                (vw, vh), vbase = cv2.getTextSize(v_label, font, font_scale, thickness)

                vx1 = max(0, ix1)
                vy1 = max(0, iy1 - vh - 6)
                vx2 = min(w - 1, vx1 + vw + 6)
                vy2 = max(vh + 4, iy1)

                cv2.rectangle(output, (vx1, vy1), (vx2, vy2), (10, 10, 10), -1)
                cv2.rectangle(output, (vx1, vy1), (vx2, vy2), box_color, 1)
                cv2.putText(
                    output,
                    v_label,
                    (vx1 + 3, vy2 - 3),
                    font,
                    font_scale,
                    box_color,
                    thickness,
                    cv2.LINE_AA,
                )

        # 4. Render Summary Overlay HUD
        if self.config.draw_summary_hud:
            total = len(bay_states)
            occ = sum(1 for s in bay_states.values() if s.current_state == OccupancyState.OCCUPIED)
            vac = sum(1 for s in bay_states.values() if s.current_state == OccupancyState.VACANT)
            unk = sum(1 for s in bay_states.values() if s.current_state == OccupancyState.UNKNOWN)
            occld = sum(1 for s in bay_states.values() if s.current_state == OccupancyState.OCCLUDED)

            hud_text = f"Total {total}  |  Occupied {occ}  |  Vacant {vac}  |  Unknown {unk}"
            if occld > 0:
                hud_text += f"  |  Occluded {occld}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), tbase = cv2.getTextSize(hud_text, font, font_scale, thickness)

            # Draw HUD card at top-left
            cv2.rectangle(output, (16, 16), (28 + tw, 28 + th + 10), (15, 15, 15), -1)
            cv2.rectangle(output, (16, 16), (28 + tw, 28 + th + 10), (0, 200, 255), 1)
            cv2.putText(
                output,
                hud_text,
                (22, 22 + th),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            # Render timestamp / frame pill
            time_text = f"Frame {frame_idx:04d} ({timestamp_sec:.2f}s)"
            (tim_w, tim_h), _ = cv2.getTextSize(time_text, font, 0.45, 1)
            cv2.rectangle(output, (w - tim_w - 28, 16), (w - 16, 28 + tim_h + 6), (15, 15, 15), -1)
            cv2.putText(
                output,
                time_text,
                (w - tim_w - 22, 22 + tim_h),
                font,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return output


class FFmpegStreamEncoder:
    """
    Safe streaming FFmpeg pipe encoder with concurrent stderr draining and cooperative cancellation.
    Avoids holding entire video frames in RAM by streaming frame-by-frame directly to FFmpeg stdin.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        timeout_seconds: float = 180.0,
        cancellation_event: Optional[threading.Event] = None,
    ) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_seconds = timeout_seconds
        self.cancellation_event = cancellation_event

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_out = output_path.with_suffix(".tmp.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(self.tmp_out),
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        self._stderr_lines: collections.deque[str] = collections.deque(maxlen=100)
        self._drain_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drain_thread.start()
        self._frame_count = 0
        self._closed = False

    def _drain_stderr(self) -> None:
        """Concurrently drain FFmpeg stderr to prevent pipe buffer deadlock."""
        if self.proc.stderr is None:
            return
        try:
            for line in iter(self.proc.stderr.readline, b""):
                if line:
                    decoded = line.decode("utf-8", errors="ignore").strip()
                    self._stderr_lines.append(decoded)
        except Exception:
            pass

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        """Write a single validated BGR frame to FFmpeg stdin pipe."""
        if self._closed:
            raise RuntimeError("Cannot write frame to closed FFmpeg encoder.")

        if self.cancellation_event and self.cancellation_event.is_set():
            self.cleanup()
            raise RuntimeError("FFmpeg encoding cancelled by user request.")

        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("Frame must be a numpy ndarray.")
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8, got {frame_bgr.dtype}")
        if frame_bgr.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape mismatch: expected ({self.height}, {self.width}, 3), got {frame_bgr.shape}")

        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)

        if self.proc.poll() is not None:
            err_msg = "\n".join(self._stderr_lines)
            self.cleanup()
            raise RuntimeError(f"FFmpeg process terminated early: {err_msg}")

        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(frame_bgr.tobytes())
            self._frame_count += 1
        except (BrokenPipeError, OSError) as e:
            err_msg = "\n".join(self._stderr_lines)
            self.cleanup()
            raise RuntimeError(f"FFmpeg stdin broken pipe: {e}. Stderr: {err_msg}")

    def finish(self) -> Path:
        """Close stdin, wait for encoding to finish, and return output path."""
        if self._closed:
            return self.output_path

        self._closed = True
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            self.tmp_out.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg encoding timed out after {self.timeout_seconds}s")
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"FFmpeg finish failed: {e}")

        if self.proc.returncode != 0:
            err_msg = "\n".join(self._stderr_lines)
            self.tmp_out.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg returned non-zero code {self.proc.returncode}: {err_msg}")

        # Atomically promote temporary output
        self.tmp_out.replace(self.output_path)
        return self.output_path

    def cleanup(self) -> None:
        """Clean up process and remove staging files on error or cancellation."""
        self._closed = True
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass

        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        except Exception:
            pass

        try:
            self.tmp_out.unlink(missing_ok=True)
        except Exception:
            pass


def encode_frames_to_mp4(
    frame_generator,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    total_frames: Optional[int] = None,
    timeout_seconds: float = 180.0,
    progress_callback: Optional[Any] = None,
    cancellation_event: Optional[threading.Event] = None,
) -> Path:
    """Convenience helper to encode a frame generator to MP4 using FFmpegStreamEncoder."""
    encoder = FFmpegStreamEncoder(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        timeout_seconds=timeout_seconds,
        cancellation_event=cancellation_event,
    )
    frame_count = 0
    try:
        for frame in frame_generator:
            encoder.write_frame(frame)
            frame_count += 1
            if progress_callback and total_frames and total_frames > 0:
                pct = min(99.0, (frame_count / total_frames) * 100.0)
                progress_callback(pct, f"Encoding frame {frame_count}/{total_frames}")
        return encoder.finish()
    except Exception:
        encoder.cleanup()
        raise
