"""
scripts.debug_calibration
============================
Visual debugging tool for A4 paper detection. Runs the SAME edge-detection
steps as core/calibration.py's detect_reference_diameter_px, but instead
of just pass/fail, draws every candidate contour it finds (with its point
count and area) onto the image and saves it -- so you can SEE exactly why
detection is succeeding or failing on a real photo.

Usage:
    python scripts/debug_calibration.py path/to/photo.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} path/to/photo.jpg")
    photo_path = Path(sys.argv[1])
    if not photo_path.exists():
        raise SystemExit(f"File not found: '{photo_path}'")

    image = np.array(Image.open(photo_path).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = image.shape[0] * image.shape[1]
    min_area = frame_area * 0.15

    output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    print(f"Image size: {image.shape[1]}x{image.shape[0]} ({frame_area} px total)")
    print(f"Minimum area needed (15%): {min_area:.0f} px")
    print(f"Total contours found: {len(contours)}")
    print()

    candidates_checked = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        candidates_checked += 1
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        is_convex = cv2.isContourConvex(approx)
        matched = len(approx) == 4 and is_convex
        color = (0, 255, 0) if matched else (0, 0, 255)
        cv2.drawContours(output, [approx], -1, color, 4)
        print(f"Candidate {candidates_checked}: area={area:.0f}px, "
              f"corners={len(approx)}, convex={is_convex} "
              f"{'<-- WOULD MATCH' if matched else ''}")

    if candidates_checked == 0:
        print("No contour was even large enough to be checked (below 15% area).")

    out_path = photo_path.parent / f"debug_{photo_path.stem}.png"
    cv2.imwrite(str(out_path), output)
    print(f"\nSaved annotated image to: {out_path}")
    print("Green = would be detected as the paper. Red = rejected (wrong corner count or not convex).")


if __name__ == "__main__":
    main()