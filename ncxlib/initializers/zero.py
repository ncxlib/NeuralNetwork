import numpy as np

from ncxlib.initializers import Initializer


class Zero(Initializer):

    @staticmethod
    def gen_W(N, d):
        return np.zeros((d, N))

    @staticmethod
    def gen_b(d):
        return np.zeros((d, 1))
