from ncxlib import util
import numpy as np

URL = "https://ncxlib.s3.us-east-1.amazonaws.com/data/mnist/ncxlib.mnist.data.gz"

def load_data(normalize=False):
    """
    Loads the MNIST dataset and optionally normalizes the input data to range [0, 1].

    Args:
        normalize (bool): Whether to normalize the input data to [0, 1].

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    data = util.load_data(URL, "mnist")

    X_train = np.array(data["X_train"])
    X_test = np.array(data["X_test"])
    y_train = np.array(data["y_train"])
    y_test = np.array(data["y_test"])

    if normalize:
        X_train = X_train / 255.0 
        X_test = X_test / 255.0

    return X_train, X_test, y_train, y_test
