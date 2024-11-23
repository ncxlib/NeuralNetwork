import numpy as np
from typing import Optional
from ncxlib.models import Model
from ncxlib.activations import Activation, Sigmoid
from ncxlib.losses import LossFunction, BinaryCrossEntropy
from ncxlib.optimizers import Optimizer, SGD


class Classifier(Model):
    def __init__(self, activation: Optional[Activation] = Sigmoid, loss_fn: Optional[LossFunction] = BinaryCrossEntropy, optimizer: Optional[Optimizer] = SGD):
        super().__init__(loss_fn)

        self.weights = None 
        self.bias = None
        self.inputs = None
        self.z = None
        self.probabilities = None
        self.activation = activation
        self.optimizer = optimizer()
    
    def calculate_weighted_sum(self, X: np.ndarray):
        pass