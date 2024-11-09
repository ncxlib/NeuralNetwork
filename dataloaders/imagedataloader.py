from dataloaders import DataLoader
from datasets import ImageDataset
from typing import Optional
from preprocessing import Preprocessor


class ImageDataLoader(DataLoader):
    def __init__(
        self,
        directory_path: str,
        shuffle=False,
        preprocessors: Optional[list[Preprocessor]] = [],
    ):
        super().__init__(shuffle, preprocessors)
        self.dataset = ImageDataset(directory_path)
        self.set_indices(len(self.dataset))

        self.preprocess()
