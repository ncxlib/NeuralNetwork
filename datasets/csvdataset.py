from datasets.dataset import Dataset
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, file_path: str):
        super().__init__(self)
        self.data = pd.read_csv(file_path)

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __len__(self):
        return len(self.data)
