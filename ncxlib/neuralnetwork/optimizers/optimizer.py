from abc import ABC, abstractmethod
import numpy as np 

class Optimizer(ABC):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    @abstractmethod
    def apply(self, W: np.ndarray, dl_dw: np.ndarray, b: np.ndarray, dl_db: np.ndarray) -> tuple[np.ndarray]:
        """
        Update model parameters using the calculated gradients.

        Parameters:
        - grads_and_vars: List of (gradient, variable) tuples, where each gradient
          corresponds to a model parameter.

        This method should be implemented by subclasses.
        """
        pass