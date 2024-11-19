import numpy as np 

class Region:
    def contains(self, point):
        raise NotImplementedError("Subclasses must implement this method.")

class Rectangle(Region):
    def __init__(self, top_left, size):
        self.top_left = top_left
        self.size = size + 1

        self.setup()

    def setup(self):
        x_coords = [self.top_left[0] + n for n in range(self.size)]
        y_coords = [self.top_left[1] - n for n in range(self.size)]

        self.points = set()
        for i in range(self.size ** 2):
            self.points.add((x_coords[i % self.size], y_coords[i // self.size]))
        
    def contains(self, point):
        x, y = point
        x_min = self.top_left[0]
        x_max = self.top_left[0] + self.size - 1
        y_max = self.top_left[1]
        y_min = self.top_left[1] - self.size + 1

        return x_min <= x <= x_max and y_min <= y <= y_max

        


    def __repr__(self):
        return f"Rectangle(top_left={self.top_left}, size={self.size - 1}x{self.size - 1})"
        

