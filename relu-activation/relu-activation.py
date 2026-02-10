import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x, dtype=float)
    result=np.maximum(0, x)
    return np.atleast_1d(result)