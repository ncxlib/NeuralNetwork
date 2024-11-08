from PIL import Image
import numpy as np
from preprocessing import Preprocessor
from datasets import Dataset
import pandas as pd

class ImageRescaler(Preprocessor):
    def __init__(self, target_size=(32, 32), original_size=(512, 512)):
        super().__init__()
        self.target_size = target_size 
        self.original_size = original_size 

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        expected_length = self.original_size[0] * self.original_size[1] * 3
        if image.size == expected_length:
            image = image.reshape((self.original_size[0], self.original_size[1], 3))
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_image = pil_image.resize(self.target_size)
        
        resized_array = np.array(resized_image).reshape(-1, 3)
        return resized_array
        
    def resize_all_images(self, dataset: Dataset) -> pd.DataFrame:
        resized_imgs = []

        data = dataset.data.copy()
        for _, row in data.iterrows():
            rgb_pixels = row['pixels']
            resized_image = self.resize_image(rgb_pixels)
            resized_imgs.append({"image_name": row['image_name'], "pixels": resized_image})

        dataframe = pd.DataFrame(resized_imgs)
        dataframe['image_name'] = dataframe['image_name'].astype("string")

        return dataframe

    def apply(self, dataset: Dataset) -> Dataset:
        dataset.data = self.resize_all_images(dataset)
        return dataset
