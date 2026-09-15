"""
core.landmarks
================
Detects hand landmarks using MediaPipe's Tasks API (`HandLandmarker`) --
NOT the legacy `mediapipe.solutions.hands` API. That legacy API has been
unmaintained since March 2023 and is confirmed broken on mediapipe>=0.10.10
on several platforms, which is inside your pinned `mediapipe>=0.10.11`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from core.exceptions import HandLandmarkError

MODEL_PATH = os.environ.get("HAND_LANDMARKER_MODEL_PATH", "weights/hand_landmarker.task")

# MediaPipe's 21-point hand landmark indices for the 5 fingertips.
_FINGERTIP_INDICES = (4, 8, 12, 16, 20)
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

_landmarker: mp_vision.HandLandmarker | None = None


@dataclass(frozen=True)
class Finger:
    """One detected fingertip, in pixel coordinates on the input image."""

    name: str
    x: float
    y: float


def _get_landmarker() -> mp_vision.HandLandmarker:
    """Lazily build and cache the HandLandmarker so the model is loaded
    once per process, not once per call -- constructing it is the
    expensive part, `.detect()` is cheap by comparison.
    """
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(MODEL_PATH):
            raise HandLandmarkError(
                f"Hand landmarker model not found at '{MODEL_PATH}'. "
                "Download hand_landmarker.task or set HAND_LANDMARKER_MODEL_PATH."
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        _landmarker = mp_vision.HandLandmarker.create_from_options(options)
    return _landmarker


def detect_hand_landmarks(image: np.ndarray) -> list[Finger] | None:
    """Detect one hand's five fingertips in `image`.

    Args:
        image: RGB uint8 frame (matches main.py's PIL-based decoding).

    Returns:
        A list of 5 `Finger` objects, thumb -> pinky, or `None` if no
        hand was detected.

    Raises:
        HandLandmarkError: the model file is missing, or MediaPipe
            itself raised while processing the frame.
    """
    landmarker = _get_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))

    try:
        result = landmarker.detect(mp_image)
    except Exception as exc:  # noqa: BLE001 -- translate any backend error
        raise HandLandmarkError(f"MediaPipe hand detection failed: {exc}") from exc

    if not result.hand_landmarks:
        return None

    height, width = image.shape[:2]
    hand = result.hand_landmarks[0]  # num_hands=1, so at most one entry

    return [
        Finger(name=name, x=hand[idx].x * width, y=hand[idx].y * height)
        for name, idx in zip(_FINGER_NAMES, _FINGERTIP_INDICES)
    ]