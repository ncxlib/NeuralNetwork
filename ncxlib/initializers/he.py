import numpy as np

from ncxlib.initializers import Initializer


class HeNormal(Initializer):

    @staticmethod
    def gen_W(N, d):
        return np.random.randn(d, N) * np.sqrt(2 / N)

    @staticmethod
    def gen_b(d):
        return np.random.randn((d, 1)) * np.sqrt(2 / d)
