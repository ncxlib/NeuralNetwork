from PIL import Image

class IMG_GRAYSCALER:
    def __init__(self):
        pass

    def img_to_grayscale(self, file_path, output_path):
        img = Image.open(file_path).convert("RGB")

        grayscale_img = Image.new("L", img.size)

        for x in range(img.width):
            for y in range(img.height):
                r, g, b = img.getpixel((x, y))
                gs_formula = int(.299 * r + .587 + g + .114 * b)
                grayscale_img.putpixel((x, y), gs_formula)

        grayscale_img.save(output_path)