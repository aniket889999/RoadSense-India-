"""Tests for ROI canvas letterbox transforms, containment, margin rejection, and round-trips.

Matches the algorithms implemented in apps/web/src/lib/canvasTransform.ts.
"""

from typing import NamedTuple, Optional


class Viewport(NamedTuple):
    offset_x: float
    offset_y: float
    draw_width: float
    draw_height: float
    scale: float


def compute_contain_viewport(
    canvas_width: float,
    canvas_height: float,
    image_width: float,
    image_height: float,
) -> Viewport:
    if canvas_width <= 0 or canvas_height <= 0 or image_width <= 0 or image_height <= 0:
        return Viewport(0.0, 0.0, float(canvas_width), float(canvas_height), 1.0)

    img_aspect = image_width / image_height
    canvas_aspect = canvas_width / canvas_height

    if canvas_aspect > img_aspect:
        # Pillarbox (black bars on left & right)
        draw_height = float(canvas_height)
        draw_width = draw_height * img_aspect
        offset_x = (canvas_width - draw_width) / 2.0
        offset_y = 0.0
        scale = draw_height / image_height
    else:
        # Letterbox (black bars on top & bottom)
        draw_width = float(canvas_width)
        draw_height = draw_width / img_aspect
        offset_x = 0.0
        offset_y = (canvas_height - draw_height) / 2.0
        scale = draw_width / image_width

    return Viewport(offset_x, offset_y, draw_width, draw_height, scale)


def canvas_to_image_normalized(
    canvas_x: float,
    canvas_y: float,
    viewport: Viewport,
) -> Optional[tuple[float, float]]:
    # Reject clicks outside the active letterbox viewport (margin clicks)
    if (
        canvas_x < viewport.offset_x
        or canvas_x > viewport.offset_x + viewport.draw_width
        or canvas_y < viewport.offset_y
        or canvas_y > viewport.offset_y + viewport.draw_height
    ):
        return None

    if viewport.draw_width <= 0 or viewport.draw_height <= 0:
        return None

    norm_x = (canvas_x - viewport.offset_x) / viewport.draw_width
    norm_y = (canvas_y - viewport.offset_y) / viewport.draw_height

    clamped_x = max(0.0, min(1.0, norm_x))
    clamped_y = max(0.0, min(1.0, norm_y))
    return (clamped_x, clamped_y)


def image_normalized_to_canvas(
    norm_x: float,
    norm_y: float,
    viewport: Viewport,
) -> tuple[float, float]:
    canvas_x = viewport.offset_x + norm_x * viewport.draw_width
    canvas_y = viewport.offset_y + norm_y * viewport.draw_height
    return (canvas_x, canvas_y)


def test_landscape_image_in_wide_canvas():
    """Wide canvas with 4:3 image produces pillarbox (bars on left/right)."""
    # Canvas is 16:9 (1600x900), Image is 4:3 (800x600)
    vp = compute_contain_viewport(1600, 900, 800, 600)
    assert vp.draw_height == 900
    assert vp.draw_width == 900 * (4 / 3)  # 1200
    assert vp.offset_y == 0
    assert vp.offset_x == (1600 - 1200) / 2  # 200

    # Margin click on far left (x=50, y=450) must be rejected
    assert canvas_to_image_normalized(50, 450, vp) is None

    # Margin click on far right (x=1500, y=450) must be rejected
    assert canvas_to_image_normalized(1500, 450, vp) is None

    # Center click (x=800, y=450) must map to (0.5, 0.5)
    pt = canvas_to_image_normalized(800, 450, vp)
    assert pt is not None
    assert abs(pt[0] - 0.5) < 1e-6
    assert abs(pt[1] - 0.5) < 1e-6


def test_portrait_image_in_wide_canvas():
    """Portrait image (600x800) in wide canvas (1200x600)."""
    vp = compute_contain_viewport(1200, 600, 600, 800)
    assert vp.draw_height == 600
    assert vp.draw_width == 600 * (600 / 800)  # 450
    assert vp.offset_x == (1200 - 450) / 2  # 375
    assert vp.offset_y == 0


def test_landscape_image_in_tall_canvas():
    """Tall canvas with 16:9 image produces letterbox (bars on top/bottom)."""
    # Canvas is 800x1000, Image is 1920x1080 (16:9)
    vp = compute_contain_viewport(800, 1000, 1920, 1080)
    assert vp.draw_width == 800
    assert abs(vp.draw_height - 800 * (9 / 16)) < 1e-4  # 450
    assert vp.offset_x == 0
    assert abs(vp.offset_y - (1000 - 450) / 2) < 1e-4  # 275

    # Margin click on top margin (x=400, y=100) must be rejected
    assert canvas_to_image_normalized(400, 100, vp) is None

    # Margin click on bottom margin (x=400, y=900) must be rejected
    assert canvas_to_image_normalized(400, 900, vp) is None


def test_square_image_and_canvas():
    """Square image in square canvas fits exactly with 0 margins."""
    vp = compute_contain_viewport(500, 500, 1000, 1000)
    assert vp.offset_x == 0
    assert vp.offset_y == 0
    assert vp.draw_width == 500
    assert vp.draw_height == 500

    # Corners
    assert canvas_to_image_normalized(0, 0, vp) == (0.0, 0.0)
    assert canvas_to_image_normalized(500, 500, vp) == (1.0, 1.0)


def test_normalization_round_trip():
    """Coordinates converted to canvas and back must match within floating precision."""
    vp = compute_contain_viewport(1280, 720, 1920, 1080)
    original_points = [
        (0.0, 0.0),
        (0.25, 0.35),
        (0.5, 0.5),
        (0.85, 0.92),
        (1.0, 1.0),
    ]
    for orig_x, orig_y in original_points:
        cx, cy = image_normalized_to_canvas(orig_x, orig_y, vp)
        norm = canvas_to_image_normalized(cx, cy, vp)
        assert norm is not None
        assert abs(norm[0] - orig_x) < 1e-5
        assert abs(norm[1] - orig_y) < 1e-5
