import numpy as np
from typing import Callable
from ncxlib.neuralnetwork.classifiers import Classifier

from ncxlib.neuralnetwork.activations import Sigmoid

class LogisticRegression(Classifier):
    def __init__(self, weights : np.ndarray, bias : np.ndarray, inputs : np.ndarray):
        '''
        Initialize LogisticRegression with weights, bias, inputs.
        '''
        super().__init__(weights, bias, inputs)
        self.activation = Sigmoid()
        

    def calculate_weighted_sum(self) -> np.ndarray:
        '''
        Returns z, the weighted sum of (self.inputs * self.w.T) + self.b.T
        '''
        z = np.dot(self.inputs, self.weights.T) + self.bias.T
        return z

    def calculate_activation_fn(self, activation_fn : Callable, weighted_sum : np.ndarray) -> np.ndarray:
        '''
        Returns a'(z). Where z is the weighted sum.
        '''
        return activation_fn(weighted_sum)
        
    
    def predict(self) -> np.ndarray: 
        '''
        Returns a prediction on a binary classification
        '''
        z = self.calculate_weighted_sum()
        probabilities = self.calculate_activation_fn(self.activation.apply, z)
        return probabilities

    def classify(self, threshold: float = 0.5) -> np.ndarray:
        '''
        Classifies inputs based on predicted probabilities.
        '''
        probabilities = self.predict()
        return (probabilities > threshold).astype(int)
