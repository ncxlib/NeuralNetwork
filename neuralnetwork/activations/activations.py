import numpy as np
from neuralnetwork.utils.check import typecheck


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
        f(x) = 1 / (1 + exp(-x))

    Parameters:
    x : np.ndarray
        Numpy array containing the weighted sum of inputs.

    Returns:
    np.ndarray
        Numpy array with the sigmoid function applied element-wise.

    Raises:
        TypeError:
            If input is not a numpy array.
        ValueError:
            If input contains NaN or infinity values.
    """

    typecheck(x)
    one = np.array(1.0, dtype=x.dtype)
    return one / (one + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.
        f(x) = max(0, x)

    Parameters:
    x : np.ndarray
        Numpy array containing the weighted sum of inputs.

    Returns:
    np.ndarray
        Numpy array with the ReLU function applied element-wise.

    Raises:
        TypeError:
            If input is not a numpy array.
        ValueError:
            If input contains NaN or infinity values.
    """

    typecheck(x)
    return np.maximum(0, x)
