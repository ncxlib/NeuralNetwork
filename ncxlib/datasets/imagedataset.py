from ncxlib.datasets.dataset import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, directory_path: str, label_numeric=False):
        self.directory_path = directory_path
        self.label_numeric = label_numeric
        self.load_images()

    def load_images(self):
        data = []
        for label in os.listdir(self.directory_path):
            label_path = os.path.join(self.directory_path, label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    # if image_name.endswith(".png"):
                    path = os.path.join(label_path, image_name)
                    image = Image.open(path).convert("RGB")
                    pixels = self.get_all_pixels(image)
                    target = int(label) if self.label_numeric else label
                    data.append({"title": str(image_name), "data": np.array(pixels), "target": target})

        self.data = pd.DataFrame(data, dtype=object)
        self.data["title"] = self.data["title"].astype("string")
        if not self.label_numeric:
            self.data["target"] = self.data["target"].astype("string")

    def get_all_pixels(self, image: Image) -> np.ndarray:
        pixels = []
        for x in range(image.width):
            for y in range(image.height):
                r, g, b = image.getpixel((x, y))
                pixels.append([r, g, b])

        return np.array(pixels)
