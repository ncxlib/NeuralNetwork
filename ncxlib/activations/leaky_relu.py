import numpy as np

from ncxlib.activations.activation import Activation
from ncxlib.util.check import typecheck


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        """
        Initialize the LeakyReLU activation with a specified alpha value.

        Parameters:
        alpha : float
            The slope for x < 0. Default is 0.01.
        """
        super().__init__()
        self.alpha = alpha

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Leaky ReLU activation function.
            f(x) = x if x > 0 else alpha * x

        Parameters:
        x : np.ndarray
            Numpy array containing the weighted sum of inputs.

        Returns:
        np.ndarray
            Numpy array with the Leaky ReLU function applied element-wise.

        Raises:
            TypeError:
                If input is not a numpy array.
            ValueError:
                If input contains NaN or infinity values.
        """

        typecheck(x)
        x = x[np.newaxis, :]
        return np.where(x > 0, x, self.alpha * x)[0]

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Leaky ReLU Derivative function.
            f'(x) = 1 if x > 0 else alpha

        Parameters:
        x : np.ndarray
            Numpy array containing the weighted sum of inputs.

        Returns:
        np.ndarray
            Numpy array with the Leaky ReLU derivative applied element-wise.
        """

        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        return dx
