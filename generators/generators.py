import pandas as pd
import numpy as np
import random
import string


def random_array(shape, low=0.0, high=1.0):
    """
    Generates a random array with the given shape and values in the range [low, high).

    Parameters:
    - shape (tuple): The shape of the array to generate.
    - low (float): The lower bound of the random values.
    - high (float): The upper bound of the random values.

    Returns:
    - np.array: A random array of the specified shape and value range.
    """
    return np.random.uniform(low, high, size=shape)


def normal_distribution_array(shape, mean=0.0, std=1.0):
    """
    Generates a random array following a normal distribution.

    Parameters:
    - shape (tuple): The shape of the array to generate.
    - mean (float): The mean of the distribution.
    - std (float): The standard deviation of the distribution.

    Returns:
    - np.array: A random array drawn from a normal distribution.
    """
    return np.random.normal(mean, std, size=shape)


def integer_array(shape, low=0, high=10):
    """
    Generates an array of random integers within the specified range.

    Parameters:
    - shape (tuple): The shape of the array to generate.
    - low (int): The minimum integer value (inclusive).
    - high (int): The maximum integer value (exclusive).

    Returns:
    - np.array: An array of random integers.
    """
    return np.random.randint(low, high, size=shape)


def generate_training_data(
    num_samples=1000, num_features=2, label_ratio=0.5, random_seed=None, normalize=False
):
    """
    Generates structured data for neural network training with labels 1/0.
    The data will have a pattern where positive and negative samples are separated into clusters.

    Parameters:
    - num_samples (int): Total number of samples to generate.
    - num_features (int): Number of features for each sample.
    - label_ratio (float): Ratio of label 1 in the dataset. Should be between 0 and 1.
    - random_seed (int): Optional seed for reproducibility.

    Returns:
    - X (np.ndarray): Generated feature matrix of shape (num_samples, num_features).
    - y (np.ndarray): Generated labels of shape (num_samples,).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    num_label_1 = int(num_samples * label_ratio)
    num_label_0 = num_samples - num_label_1

    X_label_1 = (
        np.random.randn(num_label_1, num_features) + 2
    )  # Shift this cluster to distinguish
    y_label_1 = np.ones(num_label_1)

    # Cluster for label 0
    X_label_0 = (
        np.random.randn(num_label_0, num_features) - 2
    )  # Shift in the opposite direction
    y_label_0 = np.zeros(num_label_0)

    # Combine the clusters
    X = np.vstack([X_label_1, X_label_0])
    y = np.hstack([y_label_1, y_label_0])

    # Shuffle the dataset to mix labels
    shuffle_indices = np.random.permutation(num_samples)
    X, y = X[shuffle_indices], y[shuffle_indices]

    if normalize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


    return X, y


def generate_random_csv(
    file_path: str,
    num_rows: int = 100,
    num_numeric_features: int = 3,
    num_categorical_features: int = 2,
    num_classes: int = 2,
    random_seed: int = None,
    max_value: int = 100,
):
    """
    Generates a random CSV file with specified characteristics.

    Parameters:
    - file_path (str): Path to save the generated CSV file.
    - num_rows (int): Number of rows in the generated CSV.
    - num_numeric_features (int): Number of numeric features (columns).
    - num_categorical_features (int): Number of categorical features (columns).
    - num_classes (int): Number of unique classes for the target column.
    - random_seed (int): Seed for reproducibility.
    - max_value (int): Max value for each numeric
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    data = {}

    for i in range(num_numeric_features):
        data[f"num_feature_{i+1}"] = np.random.rand(num_rows) * max_value

    for i in range(num_categorical_features):
        categories = [f"cat_{j}" for j in range(random.randint(2, 5))]
        data[f"cat_feature_{i+1}"] = [
            random.choice(categories) for _ in range(num_rows)
        ]

    data["target"] = np.random.choice([i for i in range(num_classes)], num_rows)

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df

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
