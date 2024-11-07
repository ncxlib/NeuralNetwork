from datasets import Dataset 
import numpy as np
from preprocessing import Scaler

class MinMaxScaler(Scaler):
    def __init__(self):
        super().__init__()
        
    
    def apply(self, dataset: Dataset) -> Dataset:
        numeric_data = dataset.data.select_dtypes(include='number')
        self.min_values = numeric_data.min()
        self.max_values = numeric_data.max()

        data = dataset.data.copy()
        numeric_data = dataset.data.select_dtypes(include='number')
        
        data[numeric_data.columns] = (numeric_data - self.min_values) / (self.max_values - self.min_values)
        dataset.data = data 
        return dataset
        
