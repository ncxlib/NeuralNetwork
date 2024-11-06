from dataloaders import DataLoader
from datasets import CSVDataset


class CSVDataLoader(DataLoader):
    def __init__(
        self, file_path, batch_size=32, shuffle=True, transform=None, num_workers=0
    ):
        super().__init__(batch_size, shuffle, num_workers)
        self.dataset = CSVDataset(file_path, transform)
        self.set_indices(len(self.dataset))

    def __iter__(self):
        for start in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start : start + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            yield batch_data

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
