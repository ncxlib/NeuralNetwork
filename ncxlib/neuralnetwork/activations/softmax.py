import numpy as np
from ncxlib.neuralnetwork.activations import Activation 

class Softmax(Activation):
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the Softmax function to the input array.

        exp_x = exp(x - max(x))
        f(x) = exp_x / sum(exp_x)

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
        
        Returns:
          np.ndarray: The Jacobian matrix for the Softmax derivative.
          
        """
        if self.activated is None:
            self.activated = self.apply(x)
        return self.activated * (1 - self.activated)