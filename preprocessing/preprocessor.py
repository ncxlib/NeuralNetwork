from abc import ABC, abstractmethod
from datasets import Dataset


class Preprocessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        pass
