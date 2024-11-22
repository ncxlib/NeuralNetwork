import numpy as np 

def classification_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    accuracy = np.mean(np.array(predictions) == np.array(targets))
    return accuracy