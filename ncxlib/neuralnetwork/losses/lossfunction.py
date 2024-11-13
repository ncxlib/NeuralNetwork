import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    
    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError
