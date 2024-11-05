from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate=0.01):
        if learning_rate <= 0:
            raise ValueError("Learning rate should be positive.")
        self.learning_rate = learning_rate

    @abstractmethod
    def apply_gradients(self, grads_and_vars):
        """
        Update model parameters using the calculated gradients.

        Parameters:
        - grads_and_vars: List of (gradient, variable) tuples, where each gradient
          corresponds to a model parameter.

        This method should be implemented by subclasses.
        """
        pass

    def get_config(self):
        """Returns the configuration of the optimizer."""
        return {"learning_rate": self.learning_rate}
