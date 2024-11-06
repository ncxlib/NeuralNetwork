from abc import ABC, abstractmethod

class Preprocessor(ABC):
    def __init__(self):
        pass 

    @abstractmethod
    def apply(self):
        pass