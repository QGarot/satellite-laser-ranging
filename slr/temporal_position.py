from decimal import Decimal

class TemporalPosition:
    def __init__(self, time: Decimal, x: float, y: float, z: float):
        self.time = time
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return f"time: {self.time}, x: {self.x:.18e}, y: {self.y:.18e}, z: {self.z:.18e}"