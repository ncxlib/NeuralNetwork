import numpy as np
import math

def calculate_loss(self, pred_y, y_orig):
    '''
    Takes in the labels predicted and original labels.

    Calculates the loss as a mean sq error
    '''
    pred_y = np.ndarray(pred_y)
    y_origin = np.ndarray(y_orig)

    mse = np.mean((pred_y - y_origin) ** 2)
    return mse