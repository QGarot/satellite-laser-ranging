from readers.reader import Reader
from slr.temporal_position import TemporalPosition
import numpy as np

class StationPositionReader(Reader):
    def __init__(self, file: str):
        super().__init__(file)
        self.memory_map = np.memmap(self.npy_file,
                                    mode="r",
                                    shape=(self.number_of_recordings, self.number_of_columns),
                                    dtype=float)

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
