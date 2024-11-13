from ncxlib.neuralnetwork.optimizers.optimizer import Optimizer
import numpy as np
import math


class RMSProp(Optimizer):
    """
    RMS Prop optimizer updates learning rate for each parameter every time.

    E[g²] = γ * E[g²] + (1 - γ) * (∂L/∂θ)²

    θ = θ - η / sqrt(E[g²]) + ε  * (∂L/∂θ)

    Attributes:
    learning_rate : float
        The learning rate for parameter updates.
    decay_rate : float
        The decay_rate factor.

    """

    def __init__(self, learning_rate = 0.01, decay_rate = 0.9, epsilon = 1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.decay_w = 0
        self.decay_b = 0
        self.epsilon = epsilon

    def apply(self, W: np.ndarray, dl_dw: np.ndarray, b: np.ndarray, dl_db: np.ndarray) -> tuple[np.ndarray]:

        self.decay_w = self.decay_rate * self.decay_w + (1 - self.decay_rate) * np.square(dl_dw)
        self.decay_b = self.decay_rate * self.decay_b + (1 - self.decay_rate) * np.square(dl_db)
        
        learning_rate_w = self.learning_rate / ((self.decay_w ** 0.5) + self.epsilon)
        learning_rate_b = self.learning_rate / ((self.decay_b ** 0.5) + self.epsilon)

        W -= learning_rate_w * dl_dw
        b -= learning_rate_b * dl_db

        return W, b
