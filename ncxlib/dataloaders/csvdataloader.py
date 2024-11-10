from ncxlib.dataloaders import DataLoader
from ncxlib.datasets import CSVDataset
from typing import Optional
from ncxlib.preprocessing import Preprocessor


class CSVDataLoader(DataLoader):
    def __init__(
        self, file_path, shuffle=False, preprocessors: Optional[list[Preprocessor]] = []
    ):
        super().__init__(shuffle, preprocessors)
        self.dataset = CSVDataset(file_path)
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
        X = df.drop(columns=["target"]).values 
        y = df["target"].values
        return X, y