from abc import ABC
from typing import Optional

import numpy as np

from ncxlib.evaluation import (balanced_accuracy, classification_accuracy,
                               roc_area)
from ncxlib.losses import LossFunction, MeanSquaredError


class Model:
    def __init__(self, loss_fn: Optional[LossFunction] = MeanSquaredError):
        self.loss_fn = loss_fn

    def _compile(self, X: np.ndarray) -> None:
        pass

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray, multiple: Optional[bool] = True) -> np.ndarray:
        pass

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[list[str]] = ["classification"],
        show=True,
    ) -> float:
        results = {}

        predictions, probabilities = self.predict(X, multiple=True)

        for metric in metrics:
            if metric == "classification":
                accuracy = classification_accuracy(predictions, y)

            elif metric == "balanced":
                accuracy = balanced_accuracy(predictions, y)

            elif metric == "roc":
                accuracy = roc_area(probabilities, y)

            results[metric] = accuracy

        if show:
            for k, v in results.items():
                if "roc" in k:
                    try:
                        print(f"{k.capitalize()}: {v[0][0]:.3f}")
                    except:
                        print(f"{k.capitalize()}: {v[0]:.3f}")

                else:
                    print(f"{k.capitalize()}: {(v * 100):.3f}%")
        return results
