import numpy as np
from core.landmarks import detect_hand_landmarks

def test_detect_hand_landmarks_empty_image():
    # Create a blank black image
    blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Should handle blank images gracefully and return empty or None
    landmarks = detect_hand_landmarks(blank_image)
    assert landmarks is None or len(landmarks) == 0
    