import numpy as np


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

import pandas as pd
import numpy as np
import random
import string

def generate_random_csv(
    file_path: str,
    num_rows: int = 100,
    num_numeric_features: int = 3,
    num_categorical_features: int = 2,
    num_classes: int = 2,
    random_seed: int = None,
    max_value: int = 100
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
        data[f"cat_feature_{i+1}"] = [random.choice(categories) for _ in range(num_rows)]
    
    data["target"] = np.random.choice([f"class_{i}" for i in range(num_classes)], num_rows)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df



