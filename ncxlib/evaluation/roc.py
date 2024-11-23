import numpy as np
from ncxlib.util import split_classes

def roc_area(probabilities, targets):
    """
    Compute the AUC, FPR, TPR, and thresholds for ROC analysis.

    Parameters:
    - probabilities (np.ndarray): Predicted probabilities.
        - Binary classification: Shape (n,).
        - Multiclass classification: Shape (n, k), where each column represents a class.
    - targets (np.ndarray): True class labels. Binary or multiclass.

    Returns:
    - auc (float): Area Under the ROC Curve.
    - fpr (np.ndarray): False Positive Rate.
    - tpr (np.ndarray): True Positive Rate.
    - thresholds (np.ndarray): Threshold values.
    """

    probabilities = probabilities.copy()
    targets = targets.copy()


    if -1 in targets:
        targets[targets == -1] = 0
        probabilities = (probabilities + 1) / 2

    if probabilities.ndim == 2:
        n_classes = probabilities.shape[1]
        aucs, fprs, tprs, threshold_list = [], [], [], []
        for class_idx in range(n_classes):
            binary_targets = (targets == class_idx).astype(int)
            class_probabilities = probabilities[:, class_idx]

            auc, fpr, tpr, thresholds = _binary_roc_area(class_probabilities, binary_targets)
            aucs.append(auc)
            fprs.append(fpr)
            tprs.append(tpr)
            threshold_list.append(thresholds)

        return aucs, fprs, tprs, threshold_list
    else:
        return _binary_roc_area(probabilities, targets)

def _binary_roc_area(probabilities, targets):
    """
    Compute AUC, FPR, TPR, and thresholds for binary classification.
    """
    probabilities = np.clip(probabilities, 0, 1)
    
    sorted_indices = np.argsort(probabilities, kind="mergesort")[::-1]
    probabilities = probabilities[sorted_indices]
    targets = targets[sorted_indices]

    tps = np.cumsum(targets)
    fps = np.cumsum(1 - targets)

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    thresholds = np.r_[np.inf, probabilities[np.where(np.diff(probabilities))]]

    auc = np.trapz(tpr, fpr)

    try:
        auc = auc[0]
    except:
        auc = auc

    return auc, fpr, tpr, thresholds
