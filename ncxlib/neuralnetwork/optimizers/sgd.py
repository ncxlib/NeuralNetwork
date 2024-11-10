from ncxlib.neuralnetwork.optimizers.optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum and Nesterov momentum.

    Attributes:
    learning_rate : float
        The learning rate for parameter updates.

    momentum : float
        The momentum factor, where 0 is vanilla gradient descent. Default is 0.

    """

    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate=learning_rate)

        if not isinstance(momentum, float) or not (0 <= momentum <= 1):
            raise ValueError("`momentum` must be a float between [0, 1].")

        self.momentum = momentum
        self.velocity = None

    def apply_gradients(self, grads_and_vars):
        """
        Applies gradients to variables to update parameters.

        Parameters:
        grads_and_vars : list of tuples
            A list of (gradient, variable) pairs where:
            - gradient (np.ndarray): The computed gradient for the variable.
            - variable (np.ndarray): The model parameter to be updated.
        """
        if self.velocity is None:
            self.velocity = [np.zeros_like(var) for _, var in grads_and_vars]

        updated_vars = []
        for i, (grad, var) in enumerate(grads_and_vars):
            if self.momentum:
                self.velocity[i] = (
                    self.momentum * self.velocity[i] - self.learning_rate * grad
                )
                updated_var = var + self.velocity[i]
            else:
                # Direct gradient descent update without momentum
                updated_var = var - self.learning_rate * grad

            # Ensure the updated variable retains the correct shape
            updated_vars.append(np.array(updated_var))

        return updated_vars

    def get_config(self):
        """
        Returns the configuration of the optimizer as a dictionary.

        Returns:
        dict
            Dictionary containing optimizer configuration, including learning rate, momentum, and nesterov.
        """
        config = super().get_config()
        config.update({"momentum": self.momentum})
        return config
