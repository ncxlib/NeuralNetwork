from abc import ABC, abstractmethod
import numpy as np
from typing import Callable

class Classifier(ABC):
    def __init__(self, weights : np.ndarray , bias : np.ndarray, inputs : np.ndarray):
        super().__init__()

        self.weights = weights
        self.bias = bias
        self.inputs  = inputs
    
    @abstractmethod
    def calculate_weighted_sum(self):
        '''
        calculate weighted sum (z)
        '''
        pass
        
    @abstractmethod
    def calculate_activation_fn(self, activation_fn : Callable, weighted_sum):
        '''
        Returns the applied activation_fn across weigthed sum
        '''
        pass
        

    