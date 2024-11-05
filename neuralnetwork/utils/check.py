import numpy as np


def typecheck(x):
    if np.isscalar(x):
        x = np.array([x], dtype=np.float64)

    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input must be a numpy array, got {type(x)} instead.")

    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinity values.")

    if not np.issubdtype(x.dtype, np.number):
        raise TypeError(
            f"Input array must be of a numeric type, got {x.dtype} instead."
        )
    return x
