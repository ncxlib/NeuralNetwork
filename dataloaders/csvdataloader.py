from dataloaders import DataLoader
from datasets import CSVDataset
from typing import Optional
from preprocessing import Preprocessor


class CSVDataLoader(DataLoader):
    def __init__(
        self, file_path, shuffle=False, preprocessors: Optional[list[Preprocessor]] = []
    ):
        super().__init__(shuffle, preprocessors)
        self.dataset = CSVDataset(file_path)
        self.set_indices(len(self.dataset))

        self.preprocess()
