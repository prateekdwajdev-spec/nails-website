from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import core.landmarks as landmarks_module
from core.landmarks import detect_hand_landmarks, _FINGER_NAMES, _FINGERTIP_INDICES
from core.exceptions import HandLandmarkError


@pytest.fixture(autouse=True)
def reset_landmarker_singleton(monkeypatch):
    """Every test gets a clean module-level `_landmarker` cache, so one
    test's mock doesn't leak into the next.
    """
    monkeypatch.setattr(landmarks_module, "_landmarker", None)


def _fake_landmark_list() -> list[SimpleNamespace]:
    """21 duck-typed landmarks; only the 5 fingertip indices get distinct,
    identifiable normalized positions -- the other 16 are unused by
    `detect_hand_landmarks` and left at a placeholder value.
    """
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    for i, idx in enumerate(_FINGERTIP_INDICES):
        landmarks[idx] = SimpleNamespace(x=0.1 * (i + 1), y=0.2 * (i + 1))
    return landmarks


def test_detect_hand_landmarks_returns_five_fingers(monkeypatch, synthetic_hand_image):
    fake_result = SimpleNamespace(hand_landmarks=[_fake_landmark_list()])
    fake_landmarker = MagicMock()
    fake_landmarker.detect.return_value = fake_result
    monkeypatch.setattr(landmarks_module, "_get_landmarker", lambda: fake_landmarker)

    fingers = detect_hand_landmarks(synthetic_hand_image)

    assert fingers is not None
    assert len(fingers) == 5
    assert [f.name for f in fingers] == list(_FINGER_NAMES)

    height, width = synthetic_hand_image.shape[:2]
    # index finger is the 2nd fingertip (i=1) -> normalized (0.2, 0.4)
    index_finger = fingers[1]
    assert index_finger.x == pytest.approx(0.2 * width)
    assert index_finger.y == pytest.approx(0.4 * height)


def test_detect_hand_landmarks_returns_none_when_no_hand_detected(monkeypatch, synthetic_hand_image):
    fake_result = SimpleNamespace(hand_landmarks=[])
    fake_landmarker = MagicMock()
    fake_landmarker.detect.return_value = fake_result
    monkeypatch.setattr(landmarks_module, "_get_landmarker", lambda: fake_landmarker)

    assert detect_hand_landmarks(synthetic_hand_image) is None


def test_detect_hand_landmarks_wraps_backend_errors(monkeypatch, synthetic_hand_image):
    fake_landmarker = MagicMock()
    fake_landmarker.detect.side_effect = RuntimeError("backend blew up")
    monkeypatch.setattr(landmarks_module, "_get_landmarker", lambda: fake_landmarker)

    with pytest.raises(HandLandmarkError):
        detect_hand_landmarks(synthetic_hand_image)


def test_get_landmarker_raises_when_model_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(landmarks_module, "MODEL_PATH", str(tmp_path / "does_not_exist.task"))

    with pytest.raises(HandLandmarkError):
        landmarks_module._get_landmarker()