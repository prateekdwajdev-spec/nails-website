import numpy as np
from core.measurement import calculate_nail_dimensions

def test_calculate_nail_dimensions_zero_scale():
    mask = np.zeros((100, 100), dtype=np.uint8)
    scale = 0.0
    
    width_mm, length_mm = calculate_nail_dimensions(mask, scale)
    assert width_mm == 0.0
    assert length_mm == 0.0