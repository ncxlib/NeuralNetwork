import numpy as np


def MSE(pred_y: np.ndarray, y_orig: np.ndarray):
    """
    Calculates the Mean Squared Error (MSE) loss.

    Parameters:
    pred_y : np.ndarray
        Predicted values.
    y_orig : np.ndarray
        Original labels.

    Returns:
    float
        MSE loss.
    """
    mse = np.mean((pred_y - y_orig) ** 2)
    return mse


def cross_entropy(pred_y: np.ndarray, y_orig: np.ndarray):
    """
    Calculates the Cross-Entropy (CE) loss for binary classification.

    Parameters:
    pred_y : np.ndarray
        Predicted probabilities (output from sigmoid).
    y_orig : np.ndarray
        Original labels (0 or 1).

    Returns:
    float
        Cross-Entropy loss.
    """
    pred_y = np.clip(pred_y, 1e-12, 1 - 1e-12)
    ce = -np.mean(y_orig * np.log(pred_y) + (1 - y_orig) * np.log(1 - pred_y))
    return ce
