from ncxlib.neuralnetwork.optimizers.optimizer import Optimizer
import numpy as np


class SGDMomentum(Optimizer):
    """
    Stochastic Gradient Descent with Momentum (SGD-Momentum) optimizer with momentum.

    v=γv+η⋅∇L/d𝜃
    θ=θ−v

    Attributes:
    learning_rate : float
        The learning rate for parameter updates.
    momentum : float
        The momentum factor, where 0 is vanilla gradient descent. Default is 0.

    """

    def __init__(self, learning_rate = 0.01, momentum = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = 0
        self.velocity_b = 0

    def apply(self, W: np.ndarray, dl_dw: np.ndarray, b: np.ndarray, dl_db: np.ndarray) -> tuple[np.ndarray]:

        self.velocity_w = self.momentum * self.velocity_w + self.learning_rate * dl_dw
        self.velocity_b = self.momentum * self.velocity_b + self.learning_rate * dl_db
        
        W -= self.velocity_w
        b -= self.velocity_b

        return W, b
