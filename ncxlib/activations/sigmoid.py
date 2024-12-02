import numpy as np

from ncxlib.activations.activation import Activation
from ncxlib.util.check import typecheck


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def apply(self, x: np.ndarray) -> np.ndarray:
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
        self.activated = one / (one + np.exp(-x))
        return self.activated

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

        self.activated = self.apply(x)
        return self.activated * (1 - self.activated)
