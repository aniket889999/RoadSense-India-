"""OpenCV frame annotation and FFmpeg encoding pipeline for parking occupancy video."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
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
      - VACANT: neon-green polygon with translucent green fill
      - OCCUPIED: red polygon with translucent red fill
      - UNKNOWN: gray polygon with translucent gray fill
      - OCCLUDED: amber polygon with translucent amber fill
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
        overlay = frame_bgr.copy()

        # 1. Render parking space polygons
        for bay_id, summary in bay_states.items():
            poly_px = bay_polygons_px.get(bay_id)
            if poly_px is None or len(poly_px) < 3:
                continue

            state = summary.current_state
            if state == OccupancyState.VACANT:
                color = tuple(self.config.vacant_color_bgr)
                alpha = self.config.vacant_fill_alpha
            elif state == OccupancyState.OCCUPIED:
                color = tuple(self.config.occupied_color_bgr)
                alpha = self.config.occupied_fill_alpha
            elif state == OccupancyState.OCCLUDED:
                color = tuple(self.config.occluded_color_bgr)
                alpha = self.config.occluded_fill_alpha
            else:
                color = tuple(self.config.unknown_color_bgr)
                alpha = self.config.unknown_fill_alpha

            # Fill polygon onto overlay
            cv2.fillPoly(overlay, [poly_px], color)

        # Alpha blend filled polygons
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

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
                # Place label near top-left of polygon
                min_x = int(np.min(poly_px[:, 0]))
                min_y = int(np.min(poly_px[:, 1]))
                label_text = f"{summary.operator_label} · {state.value}"
                if summary.confidence > 0.0 and state == OccupancyState.OCCUPIED:
                    label_text += f" · {summary.confidence:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness = 1
                (lw, lh), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                bg_y1 = max(0, min_y - lh - 6)
                bg_y2 = max(lh + 6, min_y)
                cv2.rectangle(output, (min_x, bg_y1), (min_x + lw + 8, bg_y2), (20, 20, 20), -1)
                cv2.rectangle(output, (min_x, bg_y1), (min_x + lw + 8, bg_y2), color, 1)
                cv2.putText(
                    output,
                    label_text,
                    (min_x + 4, bg_y2 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        # 3. Render vehicle bounding boxes (CYAN)
        if self.config.draw_vehicle_boxes:
            box_color = tuple(self.config.vehicle_box_bgr)
            for det in vehicle_detections:
                x1, y1, x2, y2 = [int(round(c)) for c in det.bbox_xyxy]
                cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)

                v_label = f"{det.class_name.upper()} {det.confidence:.2f}"
                if self.config.draw_track_ids and det.track_id is not None:
                    v_label = f"Vehicle #{det.track_id} · {det.confidence:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (vw, vh), vbase = cv2.getTextSize(v_label, font, font_scale, thickness)

                v_bg_y1 = max(0, y1 - vh - 6)
                v_bg_y2 = max(vh + 6, y1)
                cv2.rectangle(output, (x1, v_bg_y1), (x1 + vw + 8, v_bg_y2), (10, 10, 10), -1)
                cv2.putText(
                    output,
                    v_label,
                    (x1 + 4, v_bg_y2 - vbase - 2),
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


def encode_frames_to_mp4(
    frame_generator,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    total_frames: Optional[int] = None,
    timeout_seconds: float = 180.0,
    progress_callback: Optional[Any] = None,
) -> Path:
    """
    Stream BGR frames into FFmpeg stdin pipe to encode browser-compatible H.264 / yuv420p / +faststart MP4.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_suffix(".tmp.mp4")

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
        str(tmp_out),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frame_count = 0
    try:
        for frame in frame_generator:
            if proc.poll() is not None:
                _, err = proc.communicate()
                raise RuntimeError(f"FFmpeg exited early: {err.decode('utf-8', errors='ignore')}")

            proc.stdin.write(frame.tobytes())
            frame_count += 1
            if progress_callback and total_frames and total_frames > 0:
                pct = min(99.0, (frame_count / total_frames) * 100.0)
                progress_callback(pct, f"Encoding frame {frame_count}/{total_frames}")

        proc.stdin.close()
        proc.wait(timeout=timeout_seconds)
    except Exception as e:
        proc.kill()
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg encoding failed: {e}")

    if proc.returncode != 0:
        _, err = proc.communicate()
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg returned non-zero exit code {proc.returncode}: {err.decode('utf-8', errors='ignore')}")

    # Atomically promote output
    tmp_out.replace(output_path)
    return output_path
