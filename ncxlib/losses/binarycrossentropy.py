import numpy as np

from ncxlib.losses.lossfunction import LossFunction


class BinaryCrossEntropy(LossFunction):

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray):
        epsilon = 1e-12
        loss = y_true * np.log(y_pred + epsilon)
        loss += (1 - y_true) * np.log(1 - y_pred + epsilon)
        return np.sum(-loss)

    @staticmethod
    def compute_gradient(y_true: np.ndarray, y_pred: np.ndarray):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
