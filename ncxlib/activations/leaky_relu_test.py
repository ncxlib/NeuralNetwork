import numpy as np

from ncxlib.activations import LeakyReLU
from ncxlib.testing import TestCase


class LeakyRelUTest(TestCase):
    def test_positive_value(self):
        positive_values = np.random.random((2, 5))
        result = LeakyReLU(alpha=0.01).apply(positive_values)
        expected = positive_values
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_negative_values(self):
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = LeakyReLU(alpha=0.01).apply(negative_values)
        expected = 0.01 * negative_values
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_positive_values_higher_alpha(self):
        positive_values = np.random.random((2, 5))
        result = LeakyReLU(alpha=0.3).apply(positive_values)
        expected = positive_values
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_negative_values_higher_alpha(self):
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = LeakyReLU(alpha=0.3).apply(negative_values)
        expected = 0.3 * negative_values
        self.assertAllClose(result, expected, rtol=1e-05)

    # Derivative Tests
    def test_derivative_positive_value(self):
        positive_values = np.random.random((2, 5))
        result = LeakyReLU(alpha=0.01).derivative(positive_values)
        expected = np.ones(positive_values.shape)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_derivative_negative_values(self):
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = LeakyReLU(alpha=0.01).derivative(negative_values)
        expected = np.ones(negative_values.shape) * 0.01
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_derivative_positive_values_higher_alpha(self):
        positive_values = np.random.random((2, 5))
        result = LeakyReLU(alpha=0.3).derivative(positive_values)
        expected = np.ones(positive_values.shape)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_derivative_negative_values_higher_alpha(self):
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = LeakyReLU(alpha=0.3).derivative(negative_values)
        expected = np.ones(negative_values.shape) * 0.3
        self.assertAllClose(result, expected, rtol=1e-05)
