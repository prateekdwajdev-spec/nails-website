"""
core.measurement
===================
Turns a binary nail mask into real-world width/length in millimeters,
using the mm-per-pixel scale from `core.calibration`.

Geometry approach
-------------------
1. Extract the mask's largest external contour.
2. Compute its minimum-area rotated bounding rectangle
   (`cv2.minAreaRect`), which is invariant to how the finger is tilted
   in-frame.
3. The rectangle's shorter side is the nail's width (side-to-side); the
   longer side is its length (base-to-tip). This module doesn't receive
   a finger orientation vector, so it can't disambiguate width/length
   any other way -- this is a reasonable assumption for a roughly
   oval/rectangular nail mask, but revisit it if you start seeing
   swapped width/length on unusual mask shapes.
4. Convert both from pixels to millimeters via `scale` (mm-per-pixel).
"""

from __future__ import annotations

import cv2
import numpy as np

from core.exceptions import MeasurementError


def calculate_nail_dimensions(mask: np.ndarray, scale: float) -> tuple[float, float]:
    """Calculate a nail's real-world width and length from its mask.

    Args:
        mask: Binary mask (bool or uint8), shape (H, W), True/nonzero
            where the nail bed is.
        scale: mm-per-pixel, from `core.calibration.calculate_scaling_factor`.

    Returns:
        `(width_mm, length_mm)`, each rounded to 2 decimal places.

    Raises:
        ValueError: `scale` is not a positive mm-per-pixel value, or
            `mask` is None.
        MeasurementError: the mask has no extractable contour, or its
            contour/rotated rect is degenerate (near-zero area).
    """
    if scale is None or scale <= 0.0:
        raise ValueError("scale must be a positive mm-per-pixel value")
    if mask is None:
        raise ValueError("mask must not be None")

    mask_uint8 = (np.asarray(mask).astype(bool).astype(np.uint8)) * 255

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise MeasurementError("Nail mask contains no extractable contour.")

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 1.0:
        raise MeasurementError("Nail mask contour has near-zero area.")

    (_, _), (edge_a, edge_b), _ = cv2.minAreaRect(largest_contour)
    if edge_a < 1e-6 or edge_b < 1e-6:
        raise MeasurementError("Nail mask's rotated bounding rect is degenerate.")

    width_px, length_px = sorted((edge_a, edge_b))  # shorter = width, longer = length

    width_mm = round(width_px * scale, 2)
    length_mm = round(length_px * scale, 2)
    return width_mm, length_mm