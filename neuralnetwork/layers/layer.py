from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
from neuralnetwork.optimizers import SGD, Optimizer
from neuralnetwork.activations import ReLU, Activation
from neuralnetwork.losses import MeanSquaredError, LossFunction


class Layer(ABC):
    def __init__(
        self,
        n_inputs: Optional[int] = None,
        n_neurons: Optional[int] = None,
        activation: Optional[Activation] = ReLU,
        optimizer: Optional[Optimizer] = SGD,
        loss_fn: Optional[LossFunction] = MeanSquaredError,
        name: str = ""
    ):
        if not Callable:
            raise ValueError(
                "Missing activation function. Cannot be empty. Example: activation_fn=Relu"
            )

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.neurons = None
        self.loss_fn = loss_fn()
        self.name = name

    @abstractmethod
    def initialize_params(self, inputs):
        pass

    @abstractmethod
    def forward_propagation(self, inputs: np.ndarray, no_save: Optional[bool] = False) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagation(self, y_orig, y_pred):
        pass

    def calc_gradient_wrt_b(self, dl_dz):
        return dl_dz

    def calc_gradient_wrt_w(self, dl_dz, inputs):
        return dl_dz * inputs

    def calc_gradient_wrt_z(self, weighted_sum, y_pred, y_orig):
        # (∂L/∂y_pred):
        dl_dy = self.calc_gradient_wrt_y_pred(y_pred, y_orig)

        # (∂a/∂z)
        da_dz = self.activation.derivative(weighted_sum)

        dl_dz = da_dz * dl_dy
        return dl_dz

    def calc_gradient_wrt_y_pred(self, y_pred, y_orig):
        return 2 * (y_pred - y_orig) / self.n_inputs

    def calculate_loss(self, y_pred: np.ndarray, y_orig: np.ndarray):
        return 0
    
    def __repr__(self):
        return f"Layer {self.name}: {{\n\tN_Inputs: {self.n_inputs}\n\tN_Neurons: {self.n_neurons} \n}}"
