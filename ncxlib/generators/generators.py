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
    num_samples=1000,
    num_features=2,
    label_ratio=0.5,
    random_seed=None,
    normalize=False,
    to_csv=False,
    file_path="training_data.csv",
):
    """
    Generates structured data for neural network training with labels 1/0.
    The data will have a pattern where positive and negative samples are 
    separated into clusters.

    Parameters:
    - num_samples (int): Total number of samples to generate.
    - num_features (int): Number of features for each sample.
    - label_ratio (float): Ratio of label 1 in the dataset (0 to 1).
    - random_seed (int): Optional seed for reproducibility.
    - normalize (bool): Whether to normalize features.
    - to_csv (bool): Whether to save the data to a CSV file.
    - file_path (str): Path to save the CSV file (if to_csv=True).

    Returns:
    - X (np.ndarray): Feature matrix (num_samples, num_features).
    - y (np.ndarray): Labels (num_samples,).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    num_label_1 = int(num_samples * label_ratio)
    num_label_0 = num_samples - num_label_1

    X_label_1 = np.random.randn(num_label_1, num_features) + 2
    y_label_1 = np.ones(num_label_1)

    X_label_0 = np.random.randn(num_label_0, num_features) - 2
    y_label_0 = np.zeros(num_label_0)

    X = np.vstack([X_label_1, X_label_0])
    y = np.hstack([y_label_1, y_label_0])

    shuffle_indices = np.random.permutation(num_samples)
    X, y = X[shuffle_indices], y[shuffle_indices]

    if normalize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    if to_csv:
        data = {f"feature_{i+1}": X[:, i] for i in range(num_features)}
        data["target"] = y
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    return X, y
