from datasets import Dataset 
import numpy as np
from preprocessing import Scaler

class MinMaxScaler(Scaler):
    def __init__(self):
        super().__init__()
        
    def apply(self, dataset: Dataset) -> Dataset:
        data = dataset.data.copy()        
        numeric_data = data.select_dtypes(include='number')
        if not numeric_data.empty:
            self.min_values = numeric_data.min()
            self.max_values = numeric_data.max()
            data[numeric_data.columns] = (numeric_data - self.min_values) / (self.max_values - self.min_values)
        
        array_data = data.select_dtypes(include=[list, np.ndarray])
        for col in array_data.columns:
            scaled_column = []
            for x in data[col].tolist():
                x_array = np.array(x)

                if x_array.ndim == 1:
                    min_val = x_array.min()
                    max_val = x_array.max()
                    scaled_array = (x_array - min_val) / (max_val - min_val) if max_val != min_val else x_array

                elif x_array.ndim == 2:
                    min_vals = x_array.min(axis=0)
                    max_vals = x_array.max(axis=0)
                    scaled_array = (x_array - min_vals) / (max_vals - min_vals) if not np.all(max_vals == min_vals) else x_array
                
                else:
                    raise ValueError(f"Unsupported array shape {x_array.shape} in column {col}")
                
                scaled_column.append(scaled_array)
            
            data[col] = scaled_column

        dataset.data = data
        return dataset
