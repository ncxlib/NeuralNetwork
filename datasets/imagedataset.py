from datasets.dataset import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.load_images()

    def load_images(self):
        data = []
        for image_name in os.listdir(self.directory_path):
            if image_name.endswith(".png"):
                path = os.path.join(self.directory_path, image_name)
                image = Image.open(path).convert("RGB")
                pixels = self.get_all_pixels(image)
                data.append({"image_name": str(image_name), "pixels": np.array(pixels)})

        self.data = pd.DataFrame(data, dtype=object)
        self.data["image_name"] = self.data["image_name"].astype("string")

    def get_all_pixels(self, image: Image) -> np.ndarray:
        pixels = []
        for x in range(image.width):
            for y in range(image.height):
                r, g, b = image.getpixel((x, y))
                pixels.append([r, g, b])

        return np.array(pixels)
