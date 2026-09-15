import cv2
import numpy as np
import pytest

from core.calibration import calculate_scaling_factor, detect_reference_diameter_px


def _frame_from_corners(corners: np.ndarray, canvas_hw: tuple[int, int] = (800, 600)) -> np.ndarray:
    """Fill a blank canvas with a white quadrilateral at `corners`,
    standing in for a photographed A4 sheet.
    """
    canvas = np.zeros((*canvas_hw, 3), dtype=np.uint8)
    cv2.fillConvexPoly(canvas, corners.astype(np.int32), (255, 255, 255))
    return canvas


def test_calculate_scaling_factor_known_values():
    # A 210mm A4 short edge spanning 400px -> 0.525 mm/px.
    assert calculate_scaling_factor(measured_diameter_px=400.0, known_diameter_mm=210.0) == pytest.approx(0.525)


def test_calculate_scaling_factor_is_falsy_on_zero_pixels():
    assert calculate_scaling_factor(measured_diameter_px=0.0, known_diameter_mm=210.0) == 0.0


def test_detect_reference_diameter_px_finds_paper_from_fixture_corners(synthetic_paper_corners):
    # synthetic_paper_corners (from conftest.py) is a 400px x 600px
    # axis-aligned quadrilateral -> shorter edge is 400px.
    frame = _frame_from_corners(synthetic_paper_corners)
    measured_px = detect_reference_diameter_px(frame)
    assert measured_px == pytest.approx(400, rel=0.05)


def test_detect_reference_diameter_px_raises_when_no_paper_present(synthetic_hand_image):
    # synthetic_hand_image (from conftest.py) is a blank frame -- no paper in it.
    with pytest.raises(ValueError):
        detect_reference_diameter_px(synthetic_hand_image)


def test_detect_reference_diameter_px_raises_when_paper_too_small():
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    frame[10:30, 10:30] = 255  # tiny square, far below the 15% area threshold
    with pytest.raises(ValueError):
        detect_reference_diameter_px(frame)