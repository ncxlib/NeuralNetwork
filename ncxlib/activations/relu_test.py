import numpy as np

from ncxlib.activations import ReLU
from ncxlib.testing import TestCase


class RelUTest(TestCase):
    def test_basic_positive_values(self):
        positive_values = np.random.uniform(0.1, 10, (2, 5))
        result = ReLU().apply(positive_values)
        self.assertAllClose(result, positive_values, rtol=1e-05)

    def test_basic_negative_values(self):
        negative_values = np.random.uniform(-10, -0.1, (2, 5))
        result = ReLU().apply(negative_values)
        expected = np.zeros_like(negative_values)
        print(negative_values.shape, result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_1d_array(self):
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = ReLU().apply(x_1d)
        expected_1d = np.maximum(0, x_1d)
        self.assertAllClose(result_1d, expected_1d, rtol=1e-05)

    def test_3d_array(self):
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = ReLU().apply(x_3d)
        expected_3d = np.maximum(0, x_3d)
        self.assertAllClose(result_3d, expected_3d, rtol=1e-05)

    def test_near_zero_values(self):
        # Test near zero values
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = ReLU().apply(x_zero)
        expected_zero = np.maximum(0, x_zero)
        self.assertAllClose(result_zero, expected_zero, rtol=1e-05)

    def test_large_positive_values(self):
        x_large_positive = np.random.uniform(1e4, 1e5, (2, 5))
        result_large_positive = ReLU().apply(x_large_positive)
        self.assertAllClose(result_large_positive, x_large_positive, rtol=1e-05)

    def test_large_negative_values(self):
        x_large_negative = np.random.uniform(-1e5, -1e4, (2, 5))
        result_large_negative = ReLU().apply(x_large_negative)
        expected_large_negative = np.zeros_like(x_large_negative)
        self.assertAllClose(result_large_negative, expected_large_negative, rtol=1e-05)

    # Derivative Tests
    def test_derivative_basic_positive_values(self):
        positive_values = np.random.uniform(0.1, 10, (2, 5))
        result = ReLU().derivative(positive_values)
        expected = np.ones(positive_values.shape)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_derivative_basic_negative_values(self):
        negative_values = np.random.uniform(-10, -0.1, (2, 5))
        result = ReLU().derivative(negative_values)
        expected = np.zeros(negative_values.shape)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_derivative_1d_array(self):
        x_1d = np.random.uniform(-10, 10, 5)
        result_1d = ReLU().derivative(x_1d)
        expected_1d = np.ones_like(x_1d)
        expected_1d[x_1d < 0] = 0
        self.assertAllClose(result_1d, expected_1d, rtol=1e-05)

    def test_derivative_3d_array(self):
        x_3d = np.random.uniform(-10, 10, (3, 3, 3))
        result_3d = ReLU().derivative(x_3d)
        expected_3d = np.ones_like(x_3d)
        expected_3d[x_3d < 0] = 0
        self.assertAllClose(result_3d, expected_3d, rtol=1e-05)

    def test_derivative_near_zero_values(self):
        x_zero = np.random.uniform(-1e-7, 1e-7, (2, 5))
        result_zero = ReLU().derivative(x_zero)
        expected_zero = np.ones_like(x_zero)
        expected_zero[x_zero < 0] = 0
        self.assertAllClose(result_zero, expected_zero, rtol=1e-05)

    def test_derivative_large_positive_values(self):
        x_large_positive = np.random.uniform(1e4, 1e5, (2, 5))
        result_large_positive = ReLU().derivative(x_large_positive)
        expected = np.ones_like(x_large_positive)
        self.assertAllClose(result_large_positive, expected, rtol=1e-05)

    def test_derivative_large_negative_values(self):
        x_large_negative = np.random.uniform(-1e5, -1e4, (2, 5))
        result_large_negative = ReLU().derivative(x_large_negative)
        expected_large_negative = np.zeros_like(x_large_negative)
        self.assertAllClose(result_large_negative, expected_large_negative, rtol=1e-05)
