from readers.reader import Reader
from slr.temporal_matrix import TemporalMatrix
import numpy as np

class RotaryMatrixReader(Reader):
    def __init__(self, file: str):
        super().__init__(file)
        self.t0 = self.get_temporal_rotary_matrix(0).time
        self.dt = 0.1

    def get_temporal_rotary_matrix(self, n: int) -> TemporalMatrix:
        """
        Reads the temporal rotary matrix corresponding to the index n from the npy file.
        :param n: index
        :return: temporal rotary matrix corresponding to the index n
        """
        row = self.read(n)
        # Store each data
        time = row[0]
        matrix = row[1:].reshape((3, 3))  # The next 9 values form a 3x3 matrix
        return TemporalMatrix(time, matrix)

    def get_index(self, t: float) -> int:
        """
        Returns the index corresponding to the row containing the nearest timestamp of 't'.
        :param t: target timestamp to find the closest match for.
        :return: index of the matrix with the nearest timestamp.
        """
        return np.round((t - self.t0) * 86400 / self.dt).astype(int)
