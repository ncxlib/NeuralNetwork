from ncxlib.neuralnetwork.losses.lossfunction import LossFunction
import numpy as np


class BinaryCrossEntropy(LossFunction):

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def compute_gradient(y_true: np.ndarray, y_pred: np.ndarray):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon) 
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) 

