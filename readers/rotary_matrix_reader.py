from readers.reader import Reader
from slr.temporal_matrix import TemporalMatrix

class RotaryMatrixReader(Reader):
    def __init__(self, file: str):
        super().__init__(file)

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