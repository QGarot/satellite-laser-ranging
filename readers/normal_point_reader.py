from readers.reader import Reader
from slr.normal_point import NormalPoint
from decimal import Decimal, getcontext

class NormalPointReader(Reader):
    def __init__(self, file: str):
        super().__init__(file)

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


class NormalPointReader2:
    def __init__(self, file: str):
        self.file = file
        self.normal_points = []

    def read(self) -> None:
        """
        Reads the normal point data from the file and saves the generated NormalPoint objects.
        """
        # Precision
        getcontext().prec = 23
        with open(self.file, "r") as normal_points_file:
            current_line = 0
            starting_line = 6
            for line in normal_points_file:
                # Skip header
                if current_line >= starting_line:
                    # Remove white blanks and split the string w.r.t the character ' '.
                    data = line.strip().split()

                    # Store and parse each data
                    time = Decimal(data[0])
                    range = float(data[1])
                    accuracy = float(data[2])
                    redundancy = float(data[3])
                    window = float(data[4])
                    wavelength = float(data[5])

                    # Create the corresponding normal point
                    normal_point = NormalPoint(time, range, accuracy, redundancy, window, wavelength)

                    # Save the normal point
                    self.normal_points.append(normal_point)
                current_line = current_line + 1

    def get_normal_points(self) -> list[NormalPoint]:
        """
        Gets the generated NormalPoint objects.
        the 'read()' method should be called before calling this method.
        :return: list of normal point objects corresponding to the given file.
        """
        return self.normal_points
