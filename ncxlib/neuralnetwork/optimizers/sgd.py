from ncxlib.neuralnetwork.optimizers.optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    θ=θ−η⋅∇L/dθ

    Attributes:
    learning_rate : float
        The learning rate for parameter updates.
    """

    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def apply(self, W: np.ndarray, dl_dw: np.ndarray, b: np.ndarray, dl_db: np.ndarray) -> tuple[np.ndarray]:

        W -= self.learning_rate * dl_dw
        b -= self.learning_rate * dl_db

        return W, b
