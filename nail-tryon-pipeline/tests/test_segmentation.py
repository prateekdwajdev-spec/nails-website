import numpy as np
from core.segmentation import segment_nail_region

def test_segment_nail_region_empty():
    blank_image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = segment_nail_region(blank_image)
    assert mask is not None
    assert mask.shape[:2] == blank_image.shape[:2]