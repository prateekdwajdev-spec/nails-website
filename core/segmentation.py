"""
core.segmentation
====================
Segments one nail bed per finger using MobileSAM (vit_t), prompted by
the fingertip point/box from `core.landmarks`.
"""

from __future__ import annotations

import os

import numpy as np
from mobile_sam import sam_model_registry, SamPredictor

from core.exceptions import SegmentationError
from core.landmarks import Finger

CHECKPOINT_PATH = os.environ.get("MOBILE_SAM_CHECKPOINT_PATH", "weights/mobile_sam.pt")
MODEL_TYPE = "vit_t"
DEVICE = "cpu"

BOX_HALF_EXTENT_PX = 60
MIN_PREDICTED_IOU = 0.5

_predictor: SamPredictor | None = None
_last_image_id: int | None = None


def _get_predictor() -> SamPredictor:
    """Lazily load the MobileSAM checkpoint and wrap it in a
    `SamPredictor`, once per process -- not once per request.
    """
    global _predictor
    if _predictor is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise SegmentationError(
                f"MobileSAM checkpoint not found at '{CHECKPOINT_PATH}'. "
                "Download mobile_sam.pt (see this module's docstring) or "
                "set MOBILE_SAM_CHECKPOINT_PATH."
            )
        model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        model.to(device=DEVICE)
        model.eval()
        _predictor = SamPredictor(model)
    return _predictor


def _ensure_image_loaded(predictor: SamPredictor, image: np.ndarray) -> None:
    """Call `set_image()` only when `image` differs from the last frame
    this predictor embedded.
    """
    global _last_image_id
    if _last_image_id != id(image):
        predictor.set_image(image)
        _last_image_id = id(image)


def _build_box_prompt(finger: Finger, image_shape: tuple[int, ...]) -> np.ndarray:
    """A box prompt centered on the fingertip, clipped to image bounds."""
    height, width = image_shape[:2]
    x_min = max(finger.x - BOX_HALF_EXTENT_PX, 0)
    y_min = max(finger.y - BOX_HALF_EXTENT_PX, 0)
    x_max = min(finger.x + BOX_HALF_EXTENT_PX, width - 1)
    y_max = min(finger.y + BOX_HALF_EXTENT_PX, height - 1)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def segment_nail_region(image: np.ndarray, finger: Finger | None = None) -> np.ndarray:
    """Segment one nail bed, prompted by `finger`'s fingertip point."""
    if finger is None:
        raise SegmentationError(
            "segment_nail_region requires a detected finger to prompt "
            "MobileSAM with -- got None."
        )

    predictor = _get_predictor()
    _ensure_image_loaded(predictor, image)

    point_coords = np.array([[finger.x, finger.y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    box = _build_box_prompt(finger, image.shape)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < MIN_PREDICTED_IOU:
        raise SegmentationError(
            f"Best mask for finger '{finger.name}' has predicted IoU "
            f"{best_score:.2f}, below minimum {MIN_PREDICTED_IOU:.2f}."
        )

    return masks[best_idx].astype(bool)