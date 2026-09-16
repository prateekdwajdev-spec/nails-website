import pytest
import numpy as np

@pytest.fixture
def synthetic_hand_image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def synthetic_paper_corners() -> np.ndarray:
    return np.array([[100, 100], [500, 100], [500, 700], [100, 700]], dtype=np.float32)