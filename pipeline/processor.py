import cv2
import numpy as np

from core.calibration import calculate_scaling_factor, detect_reference_diameter_px
from core.exceptions import CalibrationError, HandLandmarkError
from core.landmarks import detect_hand_landmarks
from core.measurement import calculate_nail_dimensions
from core.segmentation import segment_nail_region

A4_SHORT_EDGE_MM = 210.0


def process_nail_image(
    image: np.ndarray,
    reference_object_known_size_mm: float = A4_SHORT_EDGE_MM,
    save_visualization: bool = True,
    output_path: str = "output_visualization.jpg",
):
    """Orchestrates calibration -> landmarks -> per-finger segmentation

    -> measurement -> visualization overlay.
    """
    try:
        measured_edge_px = detect_reference_diameter_px(image)
    except ValueError as exc:
        raise CalibrationError(str(exc)) from exc

    scale = calculate_scaling_factor(
        measured_edge_px, reference_object_known_size_mm
    )
    if not scale:
        raise CalibrationError(
            "Could not determine a pixel-to-mm scale for this frame."
        )

    landmarks = detect_hand_landmarks(image)
    if not landmarks:
        raise HandLandmarkError("No hand detected in frame.")

    measurements = []
    visualization_image = image.copy()
    overlay = image.copy()

    # Distinct BGR color palettes for each finger
    colors = {
        "thumb": (255, 100, 0),
        "index": (0, 255, 0),
        "middle": (0, 165, 255),
        "ring": (255, 0, 255),
        "pinky": (0, 255, 255),
    }

    for finger in landmarks:
        mask = segment_nail_region(image, finger)
        width_mm, length_mm = calculate_nail_dimensions(mask, scale)

        measurements.append(
            {
                "finger": finger.name,
                "width_mm": width_mm,
                "length_mm": length_mm,
            }
        )

        if save_visualization and mask is not None:
            color = colors.get(finger.name, (0, 255, 0))

            # 1. Color fill inside the nail bed mask
            overlay[mask > 0] = color

            # 2. Draw rotated bounding box used for width/length
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(
                    visualization_image, [box], 0, (255, 255, 255), 2
                )

            # 3. Non-overlapping text position above fingertip
            text = f"{finger.name}: {width_mm}x{length_mm}mm"
            text_pos = (int(finger.x) - 50, max(25, int(finger.y) - 25))

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                visualization_image,
                (text_pos[0] - 2, text_pos[1] - h - 4),
                (text_pos[0] + w + 2, text_pos[1] + 4),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                visualization_image,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if save_visualization:
        final_vis = cv2.addWeighted(visualization_image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(output_path, cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))

    return {"scale": scale, "measurements": measurements}