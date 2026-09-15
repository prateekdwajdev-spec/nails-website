def calculate_scaling_factor(measured_diameter_px: float, known_diameter_mm: float) -> float:
    """Calculate the mm-to-pixel scaling factor."""
    if measured_diameter_px == 0:
        return 0.0
    return known_diameter_mm / measured_diameter_px