"""Rolling-window feature extraction (last, mean, slope per vital)."""
from collections import deque
import numpy as np

WINDOW = 10


class FeatureWindow:
    def __init__(self, window: int = WINDOW):
        self.buf = deque(maxlen=window)

    def push(self, reading: np.ndarray) -> np.ndarray:
        self.buf.append(reading)
        arr = np.asarray(self.buf)
        slope = (arr[-1] - arr[0]) / max(len(arr) - 1, 1)
        return np.concatenate([arr[-1], arr.mean(axis=0), slope])
