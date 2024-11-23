from abc import ABC, abstractmethod
import numpy as np 

class Initializer(ABC):

    @abstractmethod
    def gen_W(N, d):
        pass

    @abstractmethod
    def gen_b(d):
        pass
        


