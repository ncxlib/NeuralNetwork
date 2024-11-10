from abc import ABC, abstractmethod
import numpy as np


class DataLoader(ABC):
    def __init__(self, shuffle=True, preprocessors=[]):
        self.shuffle = shuffle
        self.indices = None
        self.preprocessors = preprocessors
        self.dataset = None

    def preprocess(self):
        for preprocessor in self.preprocessors:
            self.dataset = preprocessor.apply(self.dataset)

    def set_indices(self, dataset_length):
        self.indices = np.arange(dataset_length)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        pass