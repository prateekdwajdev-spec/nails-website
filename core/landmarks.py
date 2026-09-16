"""
core.landmarks
================
Detects hand landmarks using MediaPipe's Tasks API (`HandLandmarker`).
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

_FINGERTIP_INDICES = (4, 8, 12, 16, 20)
_DIP_INDICES = (3, 7, 11, 15, 19)
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

_landmarker: mp_vision.HandLandmarker | None = None


@dataclass(frozen=True)
class Finger:
    """One detected finger carrying both tip and DIP joint pixel coordinates."""

    name: str
    x: float
    y: float
    dip_x: float
    dip_y: float

    @property
    def tip_pt(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def dip_pt(self) -> tuple[float, float]:
        return (self.dip_x, self.dip_y)


def _get_landmarker() -> mp_vision.HandLandmarker:
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(MODEL_PATH):
            raise HandLandmarkError(
                f"Hand landmarker model not found at '{MODEL_PATH}'."
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
    """Detect 5 fingers with tip and DIP joint coordinates."""
    landmarker = _get_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))

    try:
        result = landmarker.detect(mp_image)
    except Exception as exc:
        raise HandLandmarkError(f"MediaPipe hand detection failed: {exc}") from exc

    if not result.hand_landmarks:
        return None

    height, width = image.shape[:2]
    hand = result.hand_landmarks[0]

    return [
        Finger(
            name=name,
            x=hand[tip_idx].x * width,
            y=hand[tip_idx].y * height,
            dip_x=hand[dip_idx].x * width,
            dip_y=hand[dip_idx].y * height,
        )
        for name, tip_idx, dip_idx in zip(_FINGER_NAMES, _FINGERTIP_INDICES, _DIP_INDICES)
    ]