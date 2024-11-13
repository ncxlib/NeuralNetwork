from PIL import Image
import numpy as np
from ncxlib.preprocessing import Preprocessor
from ncxlib.datasets import Dataset
import pandas as pd


class ImageRescaler(Preprocessor):
    def __init__(self, target_size=(32, 32), original_size=(512, 512), flatten=False):
        super().__init__()
        self.target_size = target_size
        self.original_size = original_size
        self.flatten = flatten

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        # Check if the image needs to be reshaped from 1D to 3D for RGB images
        if image.ndim == 1:
            image = image.reshape((self.original_size[0], self.original_size[1], 3))

        # Convert to PIL Image, resize, and convert back to numpy
        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_image = pil_image.resize(self.target_size)
        resized_array = np.array(resized_image)

        # Flatten if specified
        if self.flatten:
            if resized_array.ndim == 3:  # RGB image
                resized_array = resized_array.reshape(-1)
            elif resized_array.ndim == 2:  # Grayscale image
                resized_array = resized_array.flatten()

        return resized_array

    def resize_all_images(self, dataset: Dataset) -> pd.DataFrame:
        resized_imgs = []

        for _, row in dataset.data.iterrows():
            rgb_pixels = row["data"]
            resized_image = self.resize_image(rgb_pixels)
            resized_imgs.append(
                {"title": row["title"], "data": resized_image, "target": row["target"]}
            )

        dataframe = pd.DataFrame(resized_imgs)
        dataframe["title"] = dataframe["title"].astype("string")
        if not dataset.label_numeric:
            dataframe["target"] = dataframe["target"].astype("string")

        return dataframe

    def apply(self, dataset: Dataset) -> Dataset:
        resized_data = self.resize_all_images(dataset)
        dataset.data = resized_data
        return dataset
