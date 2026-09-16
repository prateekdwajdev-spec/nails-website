import cv2
import numpy as np


def calculate_scaling_factor(measured_diameter_px: float, known_diameter_mm: float) -> float:
    """Calculate the mm-per-pixel scaling factor from a single reference
    measurement (e.g. the known short edge of an A4 sheet). Works for any
    linear reference measurement, not just circular objects, despite the
    "diameter" naming.
    """
    if measured_diameter_px == 0:
        return 0.0
    return known_diameter_mm / measured_diameter_px


def detect_reference_diameter_px(image: np.ndarray) -> float:
    """Locate an A4 sheet of paper in `image` and return the pixel length
    of its shorter edge.

    A4's short edge is a fixed 210mm regardless of whether the sheet is
    photographed in portrait or landscape orientation, so the shorter of
    the two detected edges is always the one `calculate_scaling_factor`
    should be called with.

    Args:
        image: RGB uint8 frame (matches main.py's PIL-based decoding).

    Raises:
        ValueError: no sufficiently large paper shape was found in the frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found; could not locate the A4 reference sheet.")

    frame_area = image.shape[0] * image.shape[1]
    min_area = frame_area * 0.15  # paper must cover a meaningful part of the frame

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue

        hull = cv2.convexHull(contour)
        
        # Fit a bounding rectangle directly to the hull (bypasses corner-count validation)
        _, (edge_a, edge_b), _ = cv2.minAreaRect(hull)

        if edge_a > 0 and edge_b > 0:
            # Aspect ratio check for A4 paper (~1.414 ratio, with tolerance for arm occlusion)
            aspect_ratio = max(edge_a, edge_b) / min(edge_a, edge_b)
            if 1.1 <= aspect_ratio <= 1.8:
                return float(min(edge_a, edge_b))

    raise ValueError(
        "No valid A4 reference sheet contour was found in the frame."
    )