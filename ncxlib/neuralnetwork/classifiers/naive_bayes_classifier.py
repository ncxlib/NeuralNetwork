import numpy as np
from ncxlib.neuralnetwork.layers import Layer
from ncxlib.neuralnetwork.activations import Softmax
from typing import Optional

class NaiveBayesClassifier(Layer):
    def __init__(self, name: str = " "):
        super().__init__(None, None, activation=Softmax(), optimizer=None, loss_fn=None, name=name)
        self.class_means = None
        self.class_variances = None
        self.class_priors = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Naive Bayes model by calculating class-specific means, variances, and priors.
        """
        self.classes = np.unique(y)
        self.class_means = {cls: X[y == cls].mean(axis=0) for cls in self.classes}
        self.class_variances = {cls: X[y == cls].var(axis=0) for cls in self.classes}
        self.class_priors = {cls: np.mean(y == cls) for cls in self.classes}

    def forward_propagation(self, inputs: np.ndarray, no_save: Optional[bool] = False) -> np.ndarray:
        """
        Compute class probabilities using Gaussian likelihood and priors.

        Parameters:
        - inputs (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
        - np.ndarray: Predicted probabilities for each class.
        """
        self.inputs = inputs
        likelihoods = []

        for cls in self.classes:
            mean = self.class_means[cls]
            var = np.clip(self.class_variances[cls], a_min=1e-6, a_max=None)
            prior = np.log(self.class_priors[cls])

            # LL assuming Guassian
            likelihood = (
                -0.5 * np.sum(np.log(2 * np.pi * var))  
                - 0.5 * np.sum(((inputs - mean) ** 2) / var, axis=1)  
            )
            likelihoods.append(prior + likelihood)

        log_probs = np.array(likelihoods).T
        # probs = self.activation.apply(log_probs) 

        if not no_save:
            self.activated = log_probs 

        return log_probs

    def back_propagation(self, next_layer: Layer, learning_rate: float) -> None:
        """
        NB doesnt need back propagation. Need to override ABC class.
        """
        pass