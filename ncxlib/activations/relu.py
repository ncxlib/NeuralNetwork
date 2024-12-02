import numpy as np

from ncxlib.activations import Activation
from ncxlib.util.check import typecheck


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def apply(self, x: np.ndarray) -> np.ndarray:
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
        return np.maximum(x, np.zeros_like(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid Derivative function.
            f'(x) = f(x) * (1 - f(x))

        Parameters:
        x : np.ndarray
            Numpy array containing the weighted sum of inputs.

        Returns:
        np.ndarray
            Numpy array with the sigmoid derivative applied element-wise.
        """

        dx = np.ones_like(x)
        dx[x < 0] = 0
        return dx
