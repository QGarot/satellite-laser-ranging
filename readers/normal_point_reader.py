from readers.reader import Reader
from slr.normal_point import NormalPoint
import numpy as np

class NormalPointReader(Reader):
    def __init__(self, file: str):
        super().__init__(file)
        self.memory_map = np.memmap(self.npy_file,
                                    mode="r",
                                    shape=(self.number_of_recordings, self.number_of_columns),
                                    dtype=float)

    def get_normal_points(self, n: int) -> NormalPoint:
        """
        Reads the normal point of the index n from the npy file.
        :param n: index
        :return: normal point corresponding to the index n
        """
        row = self.read(n)
        # Store each data
        time = row[0]
        measured_range = row[1]
        accuracy = row[2]
        redundancy = row[3]
        window = row[4]
        wavelength = row[5]

        # Create the corresponding normal point
        normal_point = NormalPoint(time, measured_range, accuracy, redundancy, window, wavelength)

        return normal_point
