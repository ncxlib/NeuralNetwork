from ncxlib.neuralnetwork.activations.activation import Activation
from ncxlib.neuralnetwork.utils.check import typecheck
import numpy as np


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

        # typecheck(x)
        return np.maximum(x, 0)

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

        return np.where(x > 0, 1, 0)
