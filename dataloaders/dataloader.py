from abc import ABC, abstractmethod
import numpy as np


class DataLoader(ABC):
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = None

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def set_indices(self, dataset_length):
        self.indices = np.arange(dataset_length)
        if self.shuffle:
            np.random.shuffle(self.indices)
