import pandas as pd

from ncxlib.datasets.dataset import Dataset


class CSVDataset(Dataset):
    def __init__(self, file_path: str):
        data = pd.read_csv(file_path)
        super().__init__(data)
