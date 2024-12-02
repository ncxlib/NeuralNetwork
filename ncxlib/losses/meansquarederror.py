import numpy as np

from ncxlib.losses import LossFunction


class MeanSquaredError(LossFunction):

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def compute_gradient(y_true: np.ndarray, y_pred: np.ndarray):
        return (2 / y_true.size) * (y_pred - y_true)
