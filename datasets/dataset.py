from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        return NotImplementedError

    @abstractmethod
    def __len__(self):
        return NotImplementedError
