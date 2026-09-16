"""
scripts.measure_photo
========================
Standalone sanity check for the FULL pipeline -- calibration, landmarks,
segmentation, measurement -- run for real against one photo, with no
mocks anywhere. This is the one thing that's only ever been tested in
isolated pieces so far; run this before wiring the pipeline up through
the FastAPI endpoint.

Usage:
    python scripts/measure_photo.py path/to/hand_on_a4_paper.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, UnidentifiedImageError

from core.exceptions import BaseNailException
from pipeline.processor import process_nail_image


def load_image_as_rgb_array(path: Path) -> np.ndarray:
    """Load an image file the same way main.py does: PIL -> RGB -> ndarray."""
    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise SystemExit(f"'{path}' is not a decodable image file.") from exc
    return np.array(image)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} path/to/photo.jpg")

    photo_path = Path(sys.argv[1])
    if not photo_path.exists():
        raise SystemExit(f"File not found: '{photo_path}'")

    print(f"Loading '{photo_path}'...")
    image = load_image_as_rgb_array(photo_path)
    print(f"Image shape: {image.shape}")

    print("Running full pipeline (calibration -> landmarks -> segmentation -> measurement)...")
    try:
        result = process_nail_image(image)
    except BaseNailException as exc:
        print(f"\nPipeline failed: {type(exc).__name__}")
        print(f"  {exc}")
        sys.exit(1)

    print(f"\nScale: {result['scale']:.4f} mm/px")
    print(f"{'Finger':<10} {'Width (mm)':>12} {'Length (mm)':>12}")
    for m in result["measurements"]:
        print(f"{m['finger']:<10} {m['width_mm']:>12.2f} {m['length_mm']:>12.2f}")


if __name__ == "__main__":
    main()