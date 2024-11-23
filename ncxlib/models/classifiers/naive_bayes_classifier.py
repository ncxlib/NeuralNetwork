import numpy as np
from typing import Optional
from ncxlib.models import Model
from ncxlib.activations import Activation, Softmax
from ncxlib.optimizers import Optimizer, SGD
from ncxlib.losses import LossFunction, BinaryCrossEntropy

class NaiveBayesClassifier(Model):
    def __init__(
        self,
        activation: Optional[Activation] = Softmax(),
        optimizer: Optional[Optimizer] = SGD(),
        loss_fn: Optional[LossFunction] = BinaryCrossEntropy(),
    ):
        super().__init__(loss_fn)
        self.class_means = None
        self.class_variances = None
        self.class_priors = None
        self.activation = activation

    def _compile(self, X: np.ndarray) -> None:
        """
        Initialize placeholders for class-specific statistics.
        """
        self.class_means = {}
        self.class_variances = {}
        self.class_priors = {}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Naive Bayes model by calculating class-specific means, variances, and priors.
        """

        self._compile(X)
        self.classes = np.unique(y)
        self.class_means = {cls: X[y == cls].mean(axis=0) for cls in self.classes}
        self.class_variances = {cls: X[y == cls].var(axis=0) for cls in self.classes}
        self.class_priors = {cls: np.mean(y == cls) for cls in self.classes}

    def forward_propagation(self, inputs: np.ndarray, no_save: Optional[bool] = False) -> np.ndarray:
        likelihoods = []

        for cls in self.classes:
            mean = self.class_means[cls]
            var = np.clip(self.class_variances[cls], a_min=1e-6, a_max=None)
            prior = np.log(self.class_priors[cls])

            likelihood = (
                -0.5 * np.sum(np.log(2 * np.pi * var))
                - 0.5 * np.sum(((inputs - mean) ** 2) / var, axis=1)
            )
            likelihoods.append(prior + likelihood)

        log_probs = np.array(likelihoods).T

        if not no_save:
            self.activated = log_probs 

        return log_probs
    
    def predict(self, X: np.ndarray, multiple: Optional[bool] = True) -> np.ndarray:
        log_probs = self.forward_propagation(X)
        probabilities = self.activation.apply(log_probs)

        predictions = np.argmax(probabilities, axis=1)

        if multiple:
            return predictions, probabilities
        return predictions[0], probabilities[0]
