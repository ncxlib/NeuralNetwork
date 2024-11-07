from preprocessing import Preprocessor
import numpy as np
from datasets import Dataset
import pandas as pd
class ImageGrayscaler(Preprocessor):
    def __init__(self):
        super().__init__()

    def img_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        grayscale_image = []
        for r, g, b in image:
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale_image.append(gray)
        
        return np.array(grayscale_image)
        
    def convert_all_img_to_grayscale(self, dataset : Dataset) -> pd.DataFrame:
        imgs = []

        data = dataset.data.copy()
        for _, row in data.iterrows():
            rgb_pixels = row['pixels']
            grayscale_image = self.img_to_grayscale(rgb_pixels)
            img_array = np.array(grayscale_image)
            imgs.append({"image_name": row['image_name'], "grayscale_pixels": img_array})


        return pd.DataFrame(imgs)
    

    def apply(self, dataset: Dataset) -> Dataset:
        dataset.data = self.convert_all_img_to_grayscale(dataset)
        return dataset