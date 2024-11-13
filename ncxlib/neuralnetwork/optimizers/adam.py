from ncxlib.neuralnetwork.optimizers.optimizer import Optimizer
import numpy as np
import math


class Adam(Optimizer):
    """
    Adam Optimizer.

    Attributes:
    learning_rate : float
        The learning rate for parameter updates.
    beta_1: float
        The decay_rate for first momentum.
    beta_2: float
        The decay_rate for second momentum.
    epsilon:
        A small constant to avoid division by zero

    """

    def __init__(self, learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2 
        self.epsilon = epsilon
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = -0
        self.timestep = 0

    def apply(self, W: np.ndarray, dl_dw: np.ndarray, b: np.ndarray, dl_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.timestep += 1

        self.m_w = (self.beta_1 * self.m_w) + ((1 - self.beta_1) * dl_dw)
        self.m_b = (self.beta_1 * self.m_b) + ((1 - self.beta_1) * dl_db)

        self.v_w = (self.beta_2 * self.v_w) + ((1 - self.beta_2) * np.square(dl_dw))
        self.v_b = (self.beta_2 * self.v_b) + ((1 - self.beta_2) * np.square(dl_db))

        m_w_hat = self.m_w / (1 - (self.beta_1 ** self.timestep))
        m_b_hat = self.m_b / (1 - (self.beta_1 ** self.timestep))
        v_w_hat = self.v_w / (1 - (self.beta_2 ** self.timestep))
        v_b_hat = self.v_b / (1 - (self.beta_2 ** self.timestep))

        W -= (self.learning_rate / (np.sqrt(v_w_hat) + self.epsilon)) * m_w_hat
        b -= (self.learning_rate / (np.sqrt(v_b_hat) + self.epsilon)) * m_b_hat

        return W, b
