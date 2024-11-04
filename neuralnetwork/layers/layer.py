from abc import ABC, abstractmethod
from neuralnetwork.activations import *
from typing import Callable


class Layer(ABC):
    def __init__(self, activation_fn: Callable):
        if not Callable:
            raise ValueError(
                "Missing activation function. Cannot be empty. Example: activation_fn=Relu"
            )

        self.activation_fn = activation_fn

    @abstractmethod
    def initialize_params(self):
        pass

    @abstractmethod
    def forward_propagation(self):
        pass

    @abstractmethod
    def back_propagation(self):
        pass

    @abstractmethod
    def update_params(self):
        pass
