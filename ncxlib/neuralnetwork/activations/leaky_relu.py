from ncxlib.neuralnetwork.activations.activation import Activation
from ncxlib.neuralnetwork.utils.check import typecheck
import numpy as np

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
        return np.where(x > 0, x, self.alpha * x)

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

        
        return np.array([xi * self.alpha if xi < 0 else xi for xi in x])
