from abc import ABC
import numpy as np
import pandas as pd


class Dataset(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __len__(self):
        return len(self.data)
