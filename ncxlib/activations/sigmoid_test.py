import numpy as np

from ncxlib.activations import Sigmoid
from ncxlib.testing import TestCase


def _test_sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        z = np.exp(x)
        return z / (1 + z)

def _test_sigmoid_derivative(x):
    a = _test_sigmoid(x)
    return a * (1 - a)


class SigmoidTest(TestCase):

    def basic_test_between_0_1(self):
        x = np.random.uniform(0, 1, (2, 5))
        result = Sigmoid().apply(x)[0]
        expected = np.vectorize(_test_sigmoid)(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_1d_array(self):
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = Sigmoid().apply(x_1d)
        expected_1d = np.vectorize(_test_sigmoid)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-05)

    def test_3d_array(self):
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = Sigmoid().apply(x_3d)
        expected_3d = np.vectorize(_test_sigmoid)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-05)

    def test_near_zero_values(self):
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = Sigmoid().apply(x_zero)
        expected_zero = np.vectorize(_test_sigmoid)(x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-05)

    def test_large_positive_values(self):
        x_large_positive = np.random.uniform(10, 100, (2, 5))
        result_large_positive = Sigmoid().apply(x_large_positive)
        expected_large_positive = np.vectorize(_test_sigmoid)(x_large_positive)
        self.assertAllClose(result_large_positive, expected_large_positive, rtol=1e-05)

    def test_large_negative_values(self):
        x_large_negative = np.random.uniform(-100, -10, (2, 5))
        result_large_negative = Sigmoid().apply(x_large_negative)
        expected_large_negative = np.vectorize(_test_sigmoid)(x_large_negative)
        self.assertAllClose(result_large_negative, expected_large_negative, rtol=1e-05)

    # Derivative Tests
    def basic_derivative_test_between_0_1(self):
        x = np.random.uniform(0, 1, (2, 5))
        result = Sigmoid().derivative(x)[0]
        expected = np.vectorize(_test_sigmoid_derivative)(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_derivative_1d_array(self):
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = Sigmoid().derivative(x_1d)
        expected_1d = np.vectorize(_test_sigmoid_derivative)(x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-05)

    def test_derivative_3d_array(self):
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = Sigmoid().derivative(x_3d)
        expected_3d = np.vectorize(_test_sigmoid_derivative)(x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-05)

    def test_derivative_near_zero_values(self):
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = Sigmoid().derivative(x_zero)
        expected_zero = np.vectorize(_test_sigmoid_derivative)(x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-05)

    def test_derivative_large_positive_values(self):
        x_large_positive = np.random.uniform(10, 100, (2, 5))
        result_large_positive = Sigmoid().derivative(x_large_positive)
        expected_large_positive = np.vectorize(_test_sigmoid_derivative)(x_large_positive)
        self.assertAllClose(result_large_positive, expected_large_positive, rtol=1e-05)

    def test_derivative_large_negative_values(self):
        x_large_negative = np.random.uniform(-100, -10, (2, 5))
        result_large_negative = Sigmoid().derivative(x_large_negative)
        expected_large_negative = np.vectorize(_test_sigmoid_derivative)(x_large_negative)
        self.assertAllClose(result_large_negative, expected_large_negative, rtol=1e-05)
