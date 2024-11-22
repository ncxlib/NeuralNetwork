import numpy as np 
from ncxlib.evaluation import split_classes

def roc_area(probabilities, targets, positive_class):

    positive_class, _ = split_classes(targets)
        
    unique_vals = np.unique(probabilities)
    if set(unique_vals) == {-1, 1}:
        probabilities = (probabilities + 1) / 2
        
    probabilities = (probabilities + 1) / 2 

    targets = targets == positive_class

    sorted_indices = np.argsort(probabilities, kind="mergesort")[::-1]
    probabilities = probabilities[sorted_indices]
    targets = targets[sorted_indices]

    weight = 1.0

    distinct_indices = np.where(np.diff(probabilities))[0]
    threshold_idxs = np.r_[distinct_indices, len(targets) - 1]

    arr = (1 - targets) * weight
    out = np.cumsum(arr, axis=None, dtype=np.float64)

    tps = out[threshold_idxs]

    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    thresholds = probabilities[threshold_idxs]
    thresholds = np.r_[np.inf, thresholds]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    auc = np.trapz(fpr, tpr)

    return auc, fpr, tpr, thresholds