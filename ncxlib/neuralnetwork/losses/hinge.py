from ncxlib.neuralnetwork.losses import LossFunction
import numpy as np


class HingeLoss(LossFunction):
    """
    Hinge Loss for binary classification.

    Loss: max(0, 1 - y_true * y_pred)
    Gradient: -y_true if (1 - y_true * y_pred) > 0, else 0
    """

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray):
        # Hinge loss formula
        losses = np.maximum(0, 1 - y_true * y_pred)
        return np.mean(losses)

    @staticmethod
    def compute_gradient(y_true: np.ndarray, y_pred: np.ndarray):
        # Gradient is -y_true where loss > 0, otherwise 0
        gradients = np.where(1 - y_true * y_pred > 0, -y_true, 0)
        return gradients
