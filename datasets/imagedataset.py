from datasets.dataset import Dataset
import os
from PIL import Image
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, directory_path: str):
        self.directory_path =directory_path
        self.load_images()
    
    def load_images(self):
        data = []
        for image_name in os.listdir(self.directory_path):
            if image_name.endswith(".png"):
                path = os.path.join(self.directory_path, image_name)
                image = Image.open(path).convert('RGB')
                pixels = self.get_all_pixels(image)
                data.append({"image_name": image_name, "pixels": pixels})

            
        self.data = pd.DataFrame(data)
            
    def get_all_pixels(self, image: Image) -> list[list[float, float, float]]:
        pixels = []
        for x in range(image.width):
            for y in range(image.height):
                r, g, b = image.getpixel((x, y))
                pixels.append([r, g, b])

        return pixels
