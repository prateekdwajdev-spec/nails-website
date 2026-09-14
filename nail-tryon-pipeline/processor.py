import numpy as np
from core.calibration import calibrate_scale
from core.landmarks import detect_hand_landmarks
from core.segmentation import segment_nail_region
from core.measurement import calculate_nail_dimensions

def process_nail_image(image: np.ndarray, reference_object_known_size_mm: float = 50.0):
    """Orchestrates the core modules to process an input image."""
    landmarks = detect_hand_landmarks(image)
    scale = calibrate_scale(image, reference_object_known_size_mm)
    mask = segment_nail_region(image)
    width_mm, length_mm = calculate_nail_dimensions(mask, scale)
    
    return {
        "landmarks": landmarks,
        "scale": scale,
        "mask": mask,
        "width_mm": width_mm,
        "length_mm": length_mm
    }