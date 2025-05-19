import numpy as np

from readers.reader import Reader
from slr.temporal_position import TemporalPosition

class OrbitReader(Reader):
    def __init__(self, file: str):
        super().__init__(file)
        self.t0 = self.loaded_data[0][0]

    def get_temporal_position(self, n: int) -> TemporalPosition:
        """
        Reads the temporal position of the index n from the npy file.
        :param n: index
        :return: temporal position corresponding to the index n
        """
        row = self.read(n)
        # Store each data
        time = row[0]
        x = row[1]
        y = row[2]
        z = row[3]

        # Create the corresponding temporal position
        temporal_station_position = TemporalPosition(time, x, y, z)

        return temporal_station_position

    def get_index(self, t: float) -> int:
        """
        Returns the index corresponding to the row containing the nearest timestamp of 't'.
        :param t: target timestamp to find the closest match for.
        :return: index of the row with the nearest timestamp.
        """
        timestamps = self.loaded_data[:, 0]
        # Absolute differences
        diff = np.abs(timestamps - t)
        return np.argmin(diff)
