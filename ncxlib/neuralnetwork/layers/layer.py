from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
from ncxlib.neuralnetwork.optimizers import SGD, Optimizer
from ncxlib.neuralnetwork.activations import ReLU, Activation
from ncxlib.neuralnetwork.losses import MeanSquaredError, LossFunction


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


        # inputs remain same for all so just store in n_inputs x 1
        self.inputs = None

        # weights size:  n_neurons * n_inputs
        self.W = None  
        self.old_W = None

        # bias and weighted sums size : n_neurons x 1
        self.b = None 
        self.z = None

        # gradients size n_neurons x 1
        self.gradients = None

        # activated outputs (store for output layer only) size n_neurons x 1
        self.activated = None

    def initialize_params(self, inputs: np.ndarray):

        # inputs always get updated
        self.inputs = inputs.reshape((self.n_inputs, 1))

        # only initialize if not initialized yet, if not use previously learned values
        if self.W is None or self.b is None:
            # self.W = np.zeros((self.n_neurons, self.n_inputs))
            self.W = np.random.randn(self.n_neurons, self.n_inputs) * np.sqrt(2 / self.n_inputs) 
            self.b = np.zeros((self.n_neurons, 1))

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

    def __repr__(self):
        return f"Layer {self.name}: {{\n\tN_Inputs: {self.n_inputs}\n\tN_Neurons: {self.n_neurons} \n}}"
