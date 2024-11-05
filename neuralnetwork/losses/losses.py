import numpy as np


def calculate_loss(pred_y: np.ndarray, y_orig: np.ndarray):
    """
    Takes in the labels predicted and original labels.

    Calculates the loss as a mean sq error
    """
    mse = np.mean((pred_y - y_orig) ** 2)
    return mse
