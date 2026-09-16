import cv2
import numpy as np
import pytest

from core.measurement import calculate_nail_dimensions
from core.exceptions import MeasurementError


@pytest.fixture
def rectangular_mask() -> np.ndarray:
    mask = np.zeros((200, 200), dtype=bool)
    mask[80:110, 60:110] = True  # 30px tall x 50px wide
    return mask


def test_calculate_nail_dimensions_axis_aligned(rectangular_mask):
    width_mm, length_mm = calculate_nail_dimensions(rectangular_mask, scale=0.5)
    assert width_mm == pytest.approx(15.0, abs=1.0)
    assert length_mm == pytest.approx(25.0, abs=1.0)


def test_calculate_nail_dimensions_rotated_mask_recovers_true_dimensions():
    # Same 50x30 rectangle, rotated 30 degrees -- minAreaRect should
    # still recover the true (untilted) width/length.
    canvas = np.zeros((200, 200), dtype=np.uint8)
    box_pts = cv2.boxPoints(((100, 100), (50, 30), 30.0)).astype(np.int32)
    cv2.fillConvexPoly(canvas, box_pts, 1)
    rotated_mask = canvas.astype(bool)

    width_mm, length_mm = calculate_nail_dimensions(rotated_mask, scale=0.5)
    assert width_mm == pytest.approx(15.0, abs=1.5)
    assert length_mm == pytest.approx(25.0, abs=1.5)


def test_calculate_nail_dimensions_accepts_uint8_mask(rectangular_mask):
    # segmentation.py returns bool masks, but this should also accept a
    # uint8 0/255 mask in case any future caller passes one.
    uint8_mask = rectangular_mask.astype(np.uint8) * 255
    width_mm, length_mm = calculate_nail_dimensions(uint8_mask, scale=0.5)
    assert width_mm == pytest.approx(15.0, abs=1.0)
    assert length_mm == pytest.approx(25.0, abs=1.0)


def test_calculate_nail_dimensions_rejects_zero_scale(rectangular_mask):
    with pytest.raises(ValueError):
        calculate_nail_dimensions(rectangular_mask, scale=0.0)


def test_calculate_nail_dimensions_rejects_negative_scale(rectangular_mask):
    with pytest.raises(ValueError):
        calculate_nail_dimensions(rectangular_mask, scale=-0.5)


def test_calculate_nail_dimensions_rejects_none_mask():
    with pytest.raises(ValueError):
        calculate_nail_dimensions(None, scale=0.5)


def test_calculate_nail_dimensions_raises_on_empty_mask():
    empty_mask = np.zeros((100, 100), dtype=bool)
    with pytest.raises(MeasurementError):
        calculate_nail_dimensions(empty_mask, scale=0.5)