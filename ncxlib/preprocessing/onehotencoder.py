from ncxlib.datasets import Dataset
import numpy as np
from ncxlib.preprocessing import Preprocessor
import pandas as pd

class OneHotEncoder(Preprocessor):
    def __init__(self):
        super().__init__()

    def apply(self, dataset: Dataset) -> Dataset:
        """
        Encodes string labels in the target column to numerical values (0, 1, 2,...).

        Args:
            dataset (Dataset): The dataset to preprocess.

        Returns:
            Dataset: The preprocessed dataset with numerical labels.
        """
        df = dataset.data
        unique_labels = df[dataset.target_column].unique()
        dataset.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        df[dataset.target_column] = df[dataset.target_column].map(dataset.label_mapping)
        dataset.data = df
        return dataset