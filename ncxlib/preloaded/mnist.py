from ncxlib import util
import numpy as np

URL = "https://ncxlib.s3.us-east-1.amazonaws.com/data/mnist/ncxlib.mnist.data.gz"

def load_data():
    data = util.load_data(URL, "mnist")

    X_train = np.array(data["X_train"])
    X_test = np.array(data["X_test"])
    y_train = np.array(data["y_train"])
    y_test = np.array(data["y_test"])
    
    return X_train, X_test, y_train, y_test