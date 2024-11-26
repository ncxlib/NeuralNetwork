import unittest
from absl.testing import parameterized 


class TestCase(parameterized.TestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)