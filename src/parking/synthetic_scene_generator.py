"""Deterministic synthetic stationary parking scene generator for Phase 2C validation harness.

Generates a zero-camera-motion fixed parking lot video and matching reference image with:
- Bounded geometry (1920x1080 / 1280x720, 30 fps, ~4s, zero network downloads)
- 4 human-defined parking bays (Bay 1: Occupied->Vacant, Bay 2: Vacant->Occupied, Bay 3: Consistently Vacant, Bay 4: Occlusion/Dropout test)
- Textured stationary background that passes ORB feature stability assessment with StabilityDecision.STABLE
- Ground truth frame-level vehicle bounding boxes and expected bay state transitions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import uuid
import cv2
import numpy as np

from src.parking.occupancy_contracts import OccupancyState, VehicleDetection


@dataclass
class SyntheticParkingFixture:
    """Complete synthetic test fixture metadata and artifact paths."""
    video_path: Path
    reference_image_path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    parking_spaces: List[Dict[str, Any]]
    approach_zones: List[Dict[str, Any]]
    expected_timeline: List[Dict[str, Any]]
    expected_final_summary: Dict[str, Any]
    frame_ground_truth_detections: Dict[int, List[VehicleDetection]]
    is_synthetic: bool = True
    disclaimer: str = "SYNTHETIC TEST EVIDENCE — NOT REAL-WORLD PERFORMANCE EVIDENCE"


def _draw_parking_lot_background(width: int = 1920, height: int = 1080) -> np.ndarray:
    """
    Renders a textured, feature-rich parking lot background image with zero camera movement.
    Provides stable, trackable ORB features across the background (curbs, markings, buildings, pavement grain).
    """
    # Base asphalt color
    np.random.seed(42)
    img = np.full((height, width, 3), (60, 62, 65), dtype=np.uint8)

    # Asphalt noise/texture for ORB feature matching
    noise = np.random.randint(-12, 12, (height, width, 3), dtype=np.int16)
    textured = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = textured

    # Background horizon / wall / trees top 20%
    horizon_y = int(height * 0.25)
    # Sky / background top
    img[0:horizon_y, :] = (180, 160, 140)
    # Building wall / fence
    cv2.rectangle(img, (0, int(height * 0.12)), (width, horizon_y), (100, 110, 120), -1)
    for bx in range(50, width, 180):
        cv2.rectangle(img, (bx, int(height * 0.14)), (bx + 80, int(height * 0.22)), (50, 60, 70), -1)
        cv2.rectangle(img, (bx + 10, int(height * 0.15)), (bx + 70, int(height * 0.21)), (220, 200, 140), -1)

    # Curb line
    cv2.line(img, (0, horizon_y), (width, horizon_y), (200, 200, 200), 5)
    cv2.line(img, (0, horizon_y + 6), (width, horizon_y + 6), (40, 40, 40), 3)

    # Sidewalk strip
    sidewalk_y = int(height * 0.32)
    cv2.rectangle(img, (0, horizon_y + 8), (width, sidewalk_y), (140, 145, 150), -1)
    for sx in range(0, width, 120):
        cv2.line(img, (sx, horizon_y + 8), (sx, sidewalk_y), (100, 100, 100), 2)

    # Curb before parking bays
    cv2.line(img, (0, sidewalk_y), (width, sidewalk_y), (230, 230, 230), 4)

    # Drive aisle dashed lane line
    aisle_y = int(height * 0.85)
    for lx in range(40, width, 100):
        cv2.line(img, (lx, aisle_y), (lx + 60, aisle_y), (240, 240, 240), 4)

    # Draw 4 distinct parking bays (top row)
    # Bay 1: x in [80, 480], y in [sidewalk_y + 20, sidewalk_y + 420]
    # Bay 2: x in [540, 940]
    # Bay 3: x in [1000, 1400]
    # Bay 4: x in [1460, 1860]
    bay_y1 = sidewalk_y + 20
    bay_y2 = sidewalk_y + 440
    bay_xs = [
        (80, 480),
        (540, 940),
        (1000, 1400),
        (1460, 1860),
    ]

    for idx, (x1, x2) in enumerate(bay_xs, start=1):
        # White parking stall lines (U-shape)
        cv2.line(img, (x1, bay_y1), (x1, bay_y2), (240, 240, 240), 6)
        cv2.line(img, (x2, bay_y1), (x2, bay_y2), (240, 240, 240), 6)
        cv2.line(img, (x1, bay_y1), (x2, bay_y1), (240, 240, 240), 6)

        # Parking space number painted on asphalt
        label = f"BAY {idx:02d}"
        cv2.putText(img, label, (x1 + 100, bay_y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

    # Water drain grate / textured surface markings for additional stable inliers
    for gx in (300, 750, 1200, 1650):
        cv2.rectangle(img, (gx, height - 100), (gx + 60, height - 40), (30, 30, 30), -1)
        for gy in range(height - 95, height - 45, 10):
            cv2.line(img, (gx + 5, gy), (gx + 55, gy), (80, 80, 80), 2)

    return img


def _draw_vehicle_on_frame(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    color_bgr: Tuple[int, int, int],
    roof_color_bgr: Tuple[int, int, int],
    heading_deg: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Draws a stylized vehicle onto frame and returns its axis-aligned bounding box (x1, y1, x2, y2)."""
    x1 = int(center_x - width / 2.0)
    y1 = int(center_y - height / 2.0)
    x2 = int(center_x + width / 2.0)
    y2 = int(center_y + height / 2.0)

    # Vehicle shadow
    cv2.rectangle(frame, (x1 + 6, y1 + 8), (x2 + 8, y2 + 10), (20, 20, 20), -1)

    # Main vehicle body
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 2)

    # Windshield / Roof
    rw = int(width * 0.7)
    rh = int(height * 0.5)
    rx1 = int(center_x - rw / 2.0)
    ry1 = int(center_y - rh / 2.0)
    rx2 = int(center_x + rw / 2.0)
    ry2 = int(center_y + rh / 2.0)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (40, 45, 50), -1)  # windshield
    cv2.rectangle(frame, (rx1 + 10, ry1 + 10), (rx2 - 10, ry2 - 10), roof_color_bgr, -1)

    # Headlights / Taillights
    cv2.rectangle(frame, (x1 + 6, y1 + 2), (x1 + 24, y1 + 10), (240, 240, 220), -1)
    cv2.rectangle(frame, (x2 - 24, y1 + 2), (x2 - 6, y1 + 10), (240, 240, 220), -1)
    cv2.rectangle(frame, (x1 + 6, y2 - 10), (x1 + 24, y2 - 2), (30, 30, 220), -1)
    cv2.rectangle(frame, (x2 - 24, y2 - 10), (x2 - 6, y2 - 2), (30, 30, 220), -1)

    return float(max(0, x1)), float(max(0, y1)), float(min(frame.shape[1], x2)), float(min(frame.shape[0], y2))


def generate_synthetic_parking_fixture(
    output_dir: Path,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    duration_seconds: float = 4.0,
    space_id_prefix: Optional[str] = None,
) -> SyntheticParkingFixture:
    """
    Generates a stationary parking lot video MP4 and reference image JPEG.
    Scene animation schedule (120 frames total at 30 fps):
      - Bay 1 (b1): Vehicle A (Blue Sedan) starts parked. At frame 40, starts driving out down the aisle. Bay becomes VACANT.
      - Bay 2 (b2): Starts vacant. Vehicle B (White SUV) drives down the aisle and parks into Bay 2 at frame 60. Bay becomes OCCUPIED.
      - Bay 3 (b3): Consistently VACANT (zero vehicles across all frames).
      - Bay 4 (b4): Vehicle C (Red Hatchback) parked throughout. Frames 45-55 simulate temporary occlusion/detector dropout, triggering OCCLUDED then recovering to OCCUPIED.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int(round(fps * duration_seconds))
    prefix = space_id_prefix or f"bay_{uuid.uuid4().hex[:8]}"

    bg_img = _draw_parking_lot_background(width, height)
    ref_image_path = output_dir / "synthetic_parking_reference.jpg"
    cv2.imwrite(str(ref_image_path), bg_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    video_path = output_dir / "synthetic_parking_stationary.mp4"

    # Define normalized 4 parking spaces
    # Bay Y: [sidewalk_y + 20, sidewalk_y + 440]
    sidewalk_y = int(height * 0.32)
    y1_px = sidewalk_y + 20
    y2_px = sidewalk_y + 440
    bay_xs_px = [
        (80, 480),     # b1
        (540, 940),    # b2
        (1000, 1400),  # b3
        (1460, 1860),  # b4
    ]

    parking_spaces: List[Dict[str, Any]] = []
    approach_zones: List[Dict[str, Any]] = []

    for idx, (x1_px, x2_px) in enumerate(bay_xs_px, start=1):
        bay_id = f"{prefix}_{idx}"
        poly_norm = [
            {"x": round(x1_px / width, 4), "y": round(y1_px / height, 4)},
            {"x": round(x2_px / width, 4), "y": round(y1_px / height, 4)},
            {"x": round(x2_px / width, 4), "y": round(y2_px / height, 4)},
            {"x": round(x1_px / width, 4), "y": round(y2_px / height, 4)},
        ]
        parking_spaces.append({
            "id": bay_id,
            "operator_label": f"Bay-0{idx}",
            "space_type": "STANDARD",
            "polygon_normalized": poly_norm,
            "active": True,
        })

        # Approach zone below the bay in drive aisle
        az_y1_px = y2_px
        az_y2_px = min(height - 10, y2_px + 120)
        az_poly_norm = [
            {"x": round(x1_px / width, 4), "y": round(az_y1_px / height, 4)},
            {"x": round(x2_px / width, 4), "y": round(az_y1_px / height, 4)},
            {"x": round(x2_px / width, 4), "y": round(az_y2_px / height, 4)},
            {"x": round(x1_px / width, 4), "y": round(az_y2_px / height, 4)},
        ]
        approach_zones.append({
            "id": f"az_{idx}",
            "parking_space_id": bay_id,
            "polygon_normalized": az_poly_norm,
        })

    # Render frames to video stream via ffmpeg pipe
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(int(fps)),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(video_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_ground_truth_detections: Dict[int, List[VehicleDetection]] = {}

    veh_w = 260.0
    veh_h = 360.0

    b1_cx = (bay_xs_px[0][0] + bay_xs_px[0][1]) / 2.0
    b1_cy = (y1_px + y2_px) / 2.0

    b2_cx = (bay_xs_px[1][0] + bay_xs_px[1][1]) / 2.0
    b2_cy = (y1_px + y2_px) / 2.0

    b4_cx = (bay_xs_px[3][0] + bay_xs_px[3][1]) / 2.0
    b4_cy = (y1_px + y2_px) / 2.0

    for frame_idx in range(total_frames):
        frame = bg_img.copy()
        dets: List[VehicleDetection] = []

        # 1. Vehicle A in Bay 1: Parked frames 0-40, drives out down aisle frames 40-75
        if frame_idx < 40:
            box = _draw_vehicle_on_frame(frame, b1_cx, b1_cy, veh_w, veh_h, (180, 80, 40), (140, 60, 30))
            dets.append(VehicleDetection(
                bbox_xyxy=box,
                confidence=0.92,
                class_id=2,
                class_name="car",
                track_id=1,
            ))
        elif frame_idx <= 75:
            progress = (frame_idx - 40) / 35.0
            cur_cy = b1_cy + (height - b1_cy + 100) * progress
            cur_cx = b1_cx + (progress * 400.0)
            if cur_cy < height + 100 and cur_cx < width + 100:
                box = _draw_vehicle_on_frame(frame, cur_cx, cur_cy, veh_w, veh_h, (180, 80, 40), (140, 60, 30))
                dets.append(VehicleDetection(
                    bbox_xyxy=box,
                    confidence=0.88,
                    class_id=2,
                    class_name="car",
                    track_id=1,
                ))

        # 2. Vehicle B in Bay 2: Drives along aisle frames 10-50, pulls into Bay 2 frames 50-80, parked frames 80-120
        if 10 <= frame_idx < 50:
            progress = (frame_idx - 10) / 40.0
            cur_cx = -150 + (b2_cx + 150) * progress
            cur_cy = height * 0.85
            box = _draw_vehicle_on_frame(frame, cur_cx, cur_cy, veh_w, veh_h, (230, 230, 230), (200, 200, 200))
            dets.append(VehicleDetection(
                bbox_xyxy=box,
                confidence=0.90,
                class_id=2,
                class_name="car",
                track_id=2,
            ))
        elif 50 <= frame_idx < 80:
            progress = (frame_idx - 50) / 30.0
            cur_cx = b2_cx
            cur_cy = (height * 0.85) - ((height * 0.85) - b2_cy) * progress
            box = _draw_vehicle_on_frame(frame, cur_cx, cur_cy, veh_w, veh_h, (230, 230, 230), (200, 200, 200))
            dets.append(VehicleDetection(
                bbox_xyxy=box,
                confidence=0.94,
                class_id=2,
                class_name="car",
                track_id=2,
            ))
        elif frame_idx >= 80:
            box = _draw_vehicle_on_frame(frame, b2_cx, b2_cy, veh_w, veh_h, (230, 230, 230), (200, 200, 200))
            dets.append(VehicleDetection(
                bbox_xyxy=box,
                confidence=0.95,
                class_id=2,
                class_name="car",
                track_id=2,
            ))

        # 3. Bay 3: Always vacant (no vehicle drawn)

        # 4. Vehicle C in Bay 4: Parked frames 0-120. During frames 45-55 temporary occlusion / detection dropout
        if not (45 <= frame_idx <= 55):
            box = _draw_vehicle_on_frame(frame, b4_cx, b4_cy, veh_w, veh_h, (40, 40, 200), (30, 30, 160))
            dets.append(VehicleDetection(
                bbox_xyxy=box,
                confidence=0.91,
                class_id=2,
                class_name="car",
                track_id=3,
            ))
        else:
            # During frames 45-55, vehicle remains partially visually obscured by temporary foreground pole/obstruction
            # but detection drops out to simulate occlusion handling
            box = _draw_vehicle_on_frame(frame, b4_cx, b4_cy, veh_w, veh_h, (40, 40, 200), (30, 30, 160))
            cv2.rectangle(frame, (int(b4_cx - 40), int(b4_cy - 180)), (int(b4_cx + 40), int(b4_cy + 180)), (80, 80, 80), -1)

        frame_ground_truth_detections[frame_idx] = dets
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    # Build expected timeline transitions
    expected_timeline = [
        {"bay_id": f"{prefix}_1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "vehicle_settled"},
        {"bay_id": f"{prefix}_2", "previous_state": "UNKNOWN", "new_state": "VACANT", "trigger_reason": "consecutive_vacant"},
        {"bay_id": f"{prefix}_3", "previous_state": "UNKNOWN", "new_state": "VACANT", "trigger_reason": "consecutive_vacant"},
        {"bay_id": f"{prefix}_4", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "vehicle_settled"},
        {"bay_id": f"{prefix}_1", "previous_state": "OCCUPIED", "new_state": "VACANT", "trigger_reason": "vehicle_departed"},
        {"bay_id": f"{prefix}_4", "previous_state": "OCCUPIED", "new_state": "OCCLUDED", "trigger_reason": "vehicle_disappeared_without_departure"},
        {"bay_id": f"{prefix}_4", "previous_state": "OCCLUDED", "new_state": "OCCUPIED", "trigger_reason": "vehicle_reappeared"},
        {"bay_id": f"{prefix}_2", "previous_state": "VACANT", "new_state": "OCCUPIED", "trigger_reason": "vehicle_parked"},
    ]

    expected_final_summary = {
        "total_bays": 4,
        "occupied_count": 2,  # Bay 2 and Bay 4
        "vacant_count": 2,    # Bay 1 and Bay 3
        "unknown_count": 0,
        "occluded_count": 0,
    }

    return SyntheticParkingFixture(
        video_path=video_path,
        reference_image_path=ref_image_path,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration_seconds,
        parking_spaces=parking_spaces,
        approach_zones=approach_zones,
        expected_timeline=expected_timeline,
        expected_final_summary=expected_final_summary,
        frame_ground_truth_detections=frame_ground_truth_detections,
    )
