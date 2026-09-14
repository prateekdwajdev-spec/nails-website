import numpy as np
from pipeline.processor import process_nail_image

def test_process_nail_image_flow():
    # Create a dummy test image (e.g., 200x200 RGB image)
    dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    result = process_nail_image(dummy_image)
    
    # Verify that all expected keys are returned by the pipeline
    assert "landmarks" in result
    assert "scale" in result
    assert "mask" in result
    assert "width_mm" in result
    assert "length_mm" in result
    
    # Verify output types/values
    assert isinstance(result["width_mm"], float)
    assert isinstance(result["length_mm"], float)