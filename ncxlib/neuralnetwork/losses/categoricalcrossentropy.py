from ncxlib.neuralnetwork.losses.lossfunction import LossFunction
import numpy as np

class CategoricalCrossEntropy(LossFunction):
    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Categorical Cross-Entropy loss.

        Args:
          y_true (np.ndarray): Ground truth (one-hot encoded).
          y_pred (np.ndarray): Predicted probabilities.

        Returns:
          float: Mean categorical cross-entropy loss.
        """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0] 

    @staticmethod
    def compute_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to predictions.

        Args:
          y_true (np.ndarray): Ground truth (one-hot encoded).
          y_pred (np.ndarray): Predicted probabilities.

        Returns:
          np.ndarray: Gradient of the loss.
        """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon) 
        return -y_true / y_pred


