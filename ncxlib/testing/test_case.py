import unittest

import numpy as np
from absl.testing import parameterized


class TestCase(parameterized.TestCase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assertAllEqual(self, x1, x2, msg=None):
        self.assertEqual(len(x1), len(x2), msg=msg)
        for e1, e2 in zip(x1, x2):
            if isinstance(e1, (list, tuple)) or isinstance(e2, (list, tuple)):
                self.assertAllEqual(e1, e2, msg=msg)
            else:
                e1 = np.array(e1)
                e2 = np.array(e2)
                self.assertEqual(e1, e2, msg=msg)

    def assertAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        if not isinstance(x1, np.ndarray):
            x1 = np.array(x1)
        if not isinstance(x2, np.ndarray):
            x2 = np.array(x2)
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol, err_msg=msg)
