from ncxlib.dataloaders import DataLoader
from ncxlib.datasets import ImageDataset
from typing import Optional
from ncxlib.preprocessing import Preprocessor
import numpy as np

class ImageDataLoader(DataLoader):
    def __init__(
        self,
        directory_path: str,
        shuffle=False,
        preprocessors: Optional[list[Preprocessor]] = [],
        non_data_columns: Optional[list[str]] = ["target", "title"],
        target_column: Optional[str] = "target",
        label_numeric=False
    ):
        super().__init__(shuffle, preprocessors)
        self.dataset = ImageDataset(directory_path)
        self.dataset.target_column = target_column
        self.dataset.label_numeric=True
        self.non_data_columns = non_data_columns
        self.target_column = target_column
        self.set_indices(len(self.dataset))

        self.preprocess()

    def get_data(self):
        """
        Extracts features (X) and target (y) from the dataset.

        Returns:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target variable array.
        """
        df = self.dataset.data
        X = df.drop(columns=self.non_data_columns).values
        y = df[self.target_column].values

        X = np.array([arr[0] for arr in X]) 

        return X, y
