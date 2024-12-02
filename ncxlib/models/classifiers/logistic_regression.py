from typing import Optional

import numpy as np
from tqdm import tqdm

from ncxlib.activations import Activation, Sigmoid
from ncxlib.evaluation import classification_accuracy
from ncxlib.losses import BinaryCrossEntropy, LossFunction
from ncxlib.models.classifiers import Classifier
from ncxlib.optimizers import SGD, Optimizer


class LogisticRegression(Classifier):
    def __init__(
        self,
        activation: Optional[Activation] = Sigmoid(),
        loss_fn: Optional[LossFunction] = BinaryCrossEntropy,
        optimizer: Optional[Optimizer] = SGD,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Logistic Regression model.
        """
        super().__init__(activation, loss_fn, optimizer)
        if random_seed:
            np.random.seed(random_seed)

    def _compile(self, X: np.ndarray, learning_rate: float):
        """
        Initialize weights and bias for the model.
        """
        # self.weights = np.random.randn(1, X.shape[1])
        self.weights = np.zeros((1, X.shape[1]))
        self.bias = 0

        self.optimizer.set_learning_rate(learning_rate)
        self.learning_rate = learning_rate

    def calculate_weighted_sum(self, X: np.ndarray) -> np.ndarray:
        """
        Returns z, the weighted sum of (self.inputs * self.w.T) + self.b.T
        """

        self.z = np.dot(X, self.weights.T) + self.bias
        return self.z

    def _gradient_descent(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Perform a single step of gradient descent to update weights and bias.

        Parameters:
        - X: Input features, shape (n_samples, n_features).
        - y: True labels, shape (n_samples, 1).
        - y_pred: Predicted probabilities, shape (n_samples, 1).
        """
        n = X.shape[0]

        #  (n, 1)
        dl_da = self.loss_fn.compute_gradient(y, y_pred)

        #  (n, 1)
        da_dz = self.activation.derivative(self.z)

        # dL/dz = (dL/dy_pred) * (dy_pred/dz) = (n, 1)
        dl_dz = dl_da * da_dz  # Shape:
        # (1, d)
        dl_dw = (dl_dz.T @ X) / n

        # (1, 1)
        dl_db = np.sum(dl_dz, axis=0, keepdims=True) / n

        self.weights, self.bias = self.optimizer.apply(
            self.weights, dl_dw, self.bias, dl_db
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: Optional[int] = 100,
        learning_rate: Optional[float] = 0.01,
    ):
        """
        Train the model using gradient descent.
        """

        self._compile(X, learning_rate)

        y = y.reshape((X.shape[0], 1))

        progress = tqdm(range(epochs))

        for epoch in progress:

            self.calculate_weighted_sum(X)

            y_pred = self.activation.apply(self.z)

            loss = self.loss_fn.compute_loss(y, y_pred) / X.shape[0]

            self._gradient_descent(X, y, y_pred)

            progress.set_description(f"Epoch: {epoch} | Loss: {loss}")

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = 0.5,
        multiple: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Predict probabilities for the given inputs.
        """
        z = np.dot(X, self.weights.T) + self.bias.T
        probabilities = self.activation.apply(z)
        predictions = (probabilities > threshold).astype(int)
        return (
            (predictions, probabilities)
            if multiple
            else (predictions[0], probabilities[0])
        )
