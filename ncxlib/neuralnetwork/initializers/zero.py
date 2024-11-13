from ncxlib.neuralnetwork.initializers import Initializer
import numpy as np 

class Zero(Initializer):
    
    @staticmethod
    def gen_W(N, d):
        return np.zeros((d, N))
    
    @staticmethod
    def gen_b(d):
        return np.zeros((d, 1))