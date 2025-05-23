import numpy as np

class TemporalPosition:
    def __init__(self, time: float, x: float, y: float, z: float):
        self.time = time
        self.x = x
        self.y = y
        self.z = z
        self.vector = np.array([x, y, z])

    def __str__(self) -> str:
        return f"time: {self.time}, x: {self.x:.18e}, y: {self.y:.18e}, z: {self.z:.18e}"