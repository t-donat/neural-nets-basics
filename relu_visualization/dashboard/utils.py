import numpy as np


def linear_layer(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * x + b


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)
