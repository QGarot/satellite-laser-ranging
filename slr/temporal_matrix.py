import numpy as np

class TemporalMatrix:
    def __init__(self, time: float, matrix: np.array):
        self.time = time
        self.matrix = matrix

    def __str__(self) -> str:
        return f"TemporalMatrix(time={self.time}) =\n{self.matrix})"