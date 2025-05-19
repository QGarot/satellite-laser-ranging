from readers.reader import Reader
from slr.temporal_position import TemporalPosition

class ReducedDynamicOrbitReader(Reader):
    def __init__(self, file: str):
        super().__init__(file)

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