import numpy as np

from ncxlib.util import split_classes


def balanced_accuracy(predictions, targets):

    positive_class, negative_class = split_classes(targets)

    TP = np.sum((targets == positive_class) & (predictions == positive_class))
    TN = np.sum((targets == negative_class) & (predictions == negative_class))
    FP = np.sum((targets == negative_class) & (predictions == positive_class))
    FN = np.sum((targets == positive_class) & (predictions == negative_class))

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (sensitivity + specificity) / 2
    return accuracy
