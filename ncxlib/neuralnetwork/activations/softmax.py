import numpy as np
from ncxlib.neuralnetwork.activations import Activation 

class Softmax(Activation):
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the Softmax function to the input array.

        Args:
          x (np.ndarray): Input array.

        Returns:
          np.ndarray: Softmax output.
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.activated = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.activated

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the Softmax function.

        Args:
          x (np.ndarray): Input array (not used directly, 
                          but kept for consistency with other activations).

        Returns:
          np.ndarray: Derivative of Softmax (Jacobian matrix).
        """
        s = self.activated.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)