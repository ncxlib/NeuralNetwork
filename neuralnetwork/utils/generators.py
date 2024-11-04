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
