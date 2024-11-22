import numpy as np

def split_classes(targets: np.ndarray):
    unique_targets = np.unique(targets)
    positive_class, negative_class = 1, 0

    # for +ve / -ve classes
    if np.sum(unique_targets >= 0) != len(unique_targets):
        positive_class = unique_targets[unique_targets > 0][0]
        negative_class = unique_targets[unique_targets < 0][0]
        
    return positive_class, negative_class