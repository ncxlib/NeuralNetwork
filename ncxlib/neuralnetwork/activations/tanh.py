from ncxlib.neuralnetwork.activations.activation import Activation
from ncxlib.neuralnetwork.utils.check import typecheck
import numpy as np


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Tanh activation function.
            f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

        Parameters:
        x : np.ndarray
            Numpy array containing the weighted sum of inputs.

        Returns:
        np.ndarray
            Numpy array with the tanh function applied element-wise.

        Raises:
            TypeError:
                If input is not a numpy array.
            ValueError:
                If input contains NaN or infinity values.
        """
        typecheck(x)
        self.activated = np.tanh(x)
        return self.activated

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Tanh derivative function.
            f'(x) = 1 - f(x)^2

        Parameters:
        x : np.ndarray
            Numpy array containing the weighted sum of inputs.

        Returns:
        np.ndarray
            Numpy array with the tanh derivative applied element-wise.
        """
        self.activated = self.apply(x)
        return 1 - self.activated ** 2
