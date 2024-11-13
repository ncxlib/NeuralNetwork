from ncxlib.neuralnetwork.losses.lossfunction import LossFunction
import numpy as np

class CategoricalCrossEntropy(LossFunction):
    
    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred), axis=1).mean()

    @staticmethod
    def compute_gradient(y_true: np.ndarray, y_pred: np.ndarray):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -y_true / y_pred
    
