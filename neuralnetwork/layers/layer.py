from abc import ABC, abstractmethod
from neuralnetwork.activations import *
from typing import Callable, Optional
import numpy as np
from neuralnetwork.optimizers.optimizer import Optimizer

class Layer(ABC):
    def __init__(self, activation_fn: Callable, optimizer: Optional[Optimizer] = None):
        if not Callable:
            raise ValueError(
                "Missing activation function. Cannot be empty. Example: activation_fn=Relu"
            )

        self.activation_fn = activation_fn
        self.optimizer = optimizer

    @abstractmethod
    def initialize_params(self, inputs):
        pass

    @abstractmethod
    def forward_propagation(self):
        pass

    @abstractmethod
    def back_propagation(self, y_orig, y_pred):
        pass

    @abstractmethod
    def calc_gradient_wrt_b(self, dl_dz):
        pass

    @abstractmethod
    def calc_gradient_wrt_w(self, dl_dz, inputs):
        pass

    @abstractmethod
    def calc_gradient_wrt_z(self, weighted_sum, y_pred, y_orig):
        pass

    @abstractmethod
    def calc_gradient_wrt_y_pred(self, y_pred, y_orig):
        pass

    @abstractmethod
    def calculate_loss(self, pred_y: np.ndarray, y_orig: np.ndarray):
        pass


        
        
        
        