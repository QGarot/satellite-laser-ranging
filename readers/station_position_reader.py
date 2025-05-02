from decimal import Decimal, getcontext

from readers.reader import Reader
from slr.temporal_position import TemporalPosition

class StationPositionReader(Reader):
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

class StationPositionReader2:
    def __init__(self, file: str):
        self.file = file
        self.temporal_station_positions = []

    def read(self) -> None:
        """
        Reads the station position data from the file and saves the generated TemporalPosition objects.
        """
        # Precision
        getcontext().prec = 23
        with open(self.file, "r") as station_positions_file:
            current_line = 0
            starting_line = 6
            for line in station_positions_file:
                # Skip header
                if current_line >= starting_line:
                    # Remove white blanks and split the string w.r.t the character ' '.
                    data = line.strip().split()

                    # Store and parse each data
                    time = Decimal(data[0])
                    x = float(data[1])
                    y = float(data[2])
                    z = float(data[3])

                    # Create the corresponding temporal position
                    temporal_station_position = TemporalPosition(time, x, y, z)

                    # Save the temporal position
                    self.temporal_station_positions.append(temporal_station_position)

                current_line = current_line + 1

    def get_normal_points(self) -> list[TemporalPosition]:
        """
        Gets the generated TemporalPosition objects.
        the 'read()' method should be called before calling this method.
        :return: list of station position objects corresponding to the given file.
        """
        return self.temporal_station_positions