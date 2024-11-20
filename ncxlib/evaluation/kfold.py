import numpy as np

class KFold:
    """
    K-Fold Cross-Validation Class.

    Parameters:
    n_splits : int
        Number of folds. Must be at least 2.
    shuffle : bool
        Whether to shuffle the data before splitting into batches.
    random_state : int, optional
        Controls the randomness of the shuffle for reproducibility.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """
        Generate indices for training and validation splits.

        Parameters:
        X : array-like, shape (n_samples, ...)
            The data to split.
        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning (not used in this basic implementation).

        Yields:
        train_indices : ndarray
            The training set indices for this split.
        val_indices : ndarray
            The validation set indices for this split.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            val_indices = indices[current:current + fold_size]
            train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])
            yield train_indices, val_indices
            current += fold_size

    