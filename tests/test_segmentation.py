from unittest.mock import MagicMock

import numpy as np
import pytest

import core.segmentation as segmentation_module
from core.segmentation import segment_nail_region, _build_box_prompt, BOX_HALF_EXTENT_PX
from core.landmarks import Finger
from core.exceptions import SegmentationError


@pytest.fixture(autouse=True)
def reset_segmentation_singleton(monkeypatch):
    """Every test gets a clean predictor/embedding-cache state, so one
    test's mock (and its call counts) can't leak into the next.
    """
    monkeypatch.setattr(segmentation_module, "_predictor", None)
    monkeypatch.setattr(segmentation_module, "_last_image_id", None)


def _mock_predictor(masks: np.ndarray, scores: np.ndarray) -> MagicMock:
    predictor = MagicMock()
    predictor.predict.return_value = (masks, scores, None)
    return predictor


def test_segment_nail_region_returns_highest_confidence_mask(monkeypatch, synthetic_hand_image):
    low_conf_mask = np.zeros((480, 640), dtype=bool)
    good_mask = np.zeros((480, 640), dtype=bool)
    good_mask[100:130, 100:150] = True
    predictor = _mock_predictor(np.stack([low_conf_mask, good_mask]), np.array([0.3, 0.9]))
    monkeypatch.setattr(segmentation_module, "_get_predictor", lambda: predictor)

    finger = Finger(name="index", x=120.0, y=115.0)
    result_mask = segment_nail_region(synthetic_hand_image, finger)

    np.testing.assert_array_equal(result_mask, good_mask)
    predictor.set_image.assert_called_once()


def test_segment_nail_region_raises_when_confidence_too_low(monkeypatch, synthetic_hand_image):
    mask = np.ones((480, 640), dtype=bool)
    predictor = _mock_predictor(np.stack([mask]), np.array([0.10]))
    monkeypatch.setattr(segmentation_module, "_get_predictor", lambda: predictor)

    finger = Finger(name="thumb", x=50.0, y=50.0)
    with pytest.raises(SegmentationError):
        segment_nail_region(synthetic_hand_image, finger)


def test_segment_nail_region_raises_when_finger_is_none(synthetic_hand_image):
    with pytest.raises(SegmentationError):
        segment_nail_region(synthetic_hand_image, finger=None)


def test_set_image_called_once_across_multiple_fingers_on_same_frame(monkeypatch, synthetic_hand_image):
    mask = np.ones((480, 640), dtype=bool)
    predictor = _mock_predictor(np.stack([mask]), np.array([0.95]))
    monkeypatch.setattr(segmentation_module, "_get_predictor", lambda: predictor)

    fingers = [Finger(name=n, x=10.0 * i, y=10.0 * i) for i, n in enumerate(("thumb", "index", "middle"))]
    for finger in fingers:
        segment_nail_region(synthetic_hand_image, finger)

    # Same image object across all 3 calls -> the expensive embedding
    # step must run exactly once, not once per finger.
    predictor.set_image.assert_called_once()
    assert predictor.predict.call_count == 3


def test_set_image_called_again_for_a_genuinely_different_image(monkeypatch, synthetic_hand_image):
    mask = np.ones((480, 640), dtype=bool)
    predictor = _mock_predictor(np.stack([mask]), np.array([0.95]))
    monkeypatch.setattr(segmentation_module, "_get_predictor", lambda: predictor)

    finger = Finger(name="index", x=10.0, y=10.0)
    other_image = synthetic_hand_image.copy()  # different object identity

    segment_nail_region(synthetic_hand_image, finger)
    segment_nail_region(other_image, finger)

    assert predictor.set_image.call_count == 2


def test_build_box_prompt_clips_to_image_bounds():
    finger = Finger(name="pinky", x=5.0, y=5.0)  # near the top-left corner
    x_min, y_min, x_max, y_max = _build_box_prompt(finger, (480, 640, 3))

    assert x_min == 0.0  # clipped, since 5 - BOX_HALF_EXTENT_PX would be negative
    assert y_min == 0.0
    assert x_max == pytest.approx(5.0 + BOX_HALF_EXTENT_PX)
    assert y_max == pytest.approx(5.0 + BOX_HALF_EXTENT_PX)