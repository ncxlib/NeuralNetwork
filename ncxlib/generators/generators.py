import pandas as pd
import numpy as np

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

def generate_cartesian_uniform_data(x_range, y_range, num_samples=100, random_seed=None): 
    if not random_seed:
        np.random.seed(random_seed)

    x = np.random.uniform(x_range[0], x_range[1], size=num_samples)
    y = np.random.uniform(y_range[0], y_range[1], size=num_samples)

    return np.column_stack((x, y))

    

def _assign_labels(points, regions, p_positive_inside=0.99, p_positive_outside=0.03, random_seed=None):
  
    if random_seed is not None:
        np.random.seed(random_seed)

    labels = []
    for point in points:
        in_positive_region = any(region.contains(point) for region in regions)

        if in_positive_region:
            label = 1 if np.random.rand() < p_positive_inside else -1
        else:
            label = 1 if np.random.rand() < p_positive_outside else -1

        labels.append(label)

    return np.array(labels)

def generate_labeled_data_with_regions(n, x_range, y_range, regions, p_positive_inside=0.99, p_positive_outside=0.03, normalize=False, random_seed=None):
    """
    Generates labeled data for classification tasks.

    Parameters:
    - n (int): Number of points to generate.
    - x_range (tuple): Range (min, max) for the x-coordinate.
    - y_range (tuple): Range (min, max) for the y-coordinate.
    - regions (list of Region): List of regions defining positive areas.
    - p_positive_inside (float): Probability of label +1 inside positive regions.
    - p_positive_outside (float): Probability of label +1 outside positive regions.
    - normalize (bool): Whether to normalize the features.
    - random_seed (int): Random seed for reproducibility.

    Returns:
    - np.ndarray: Feature matrix of shape (n, 2).
    - np.ndarray: Label vector of shape (n,).
    """
    points = generate_cartesian_uniform_data(x_range, y_range, num_samples=n, random_seed=random_seed)
    labels = _assign_labels(points, regions, p_positive_inside, p_positive_outside, random_seed)

    if normalize:
        points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)

    return points, labels
