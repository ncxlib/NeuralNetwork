import numpy as np 

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Splits data into training and testing sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable array.
        test_size (float): Proportion of data to use for testing (default 0.2).
        random_state (int): Optional random seed for reproducibility.

    Returns:
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Testing feature matrix.
        y_train (np.ndarray): Training target variable array.
        y_test (np.ndarray): Testing target variable array.
    """

    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(X)
    num_test_samples = int(test_size * num_samples)

    indices = np.random.permutation(num_samples)

    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    # Split the data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def k_fold_cross_validation(X, y, k=5, random_seed=None):
    '''
    Performs a K-Fold Cross Validation on the given dataset. 
    Used for evaluating a model's performance on different subsets of the data.
    
    Args:
        X (np.ndarray): inputs .
        y (np.ndarray): Target labels.
        k (int): Number of folds.
        random_seed (int): Optional random seed.

    Returns:
        scores (list): List of scores for each fold.
        folds (list): List of (X_train, y_train, X_test, y_test) tuples for each fold.
    '''
    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.random.permutation(len(X))
    folds_indices = np.array_split(indices, k)

    scores = []
    folds = []

    for i in range(k):
        test_indices = folds_indices[i]
        train_indices = np.concatenate([fold for j, fold in enumerate(folds_indices) if j != i])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        folds.append((X_train, y_train, X_test, y_test))
        score = np.mean(y_test == y_train[:len(y_test)])  
        scores.append(score)
    
    return scores, folds
