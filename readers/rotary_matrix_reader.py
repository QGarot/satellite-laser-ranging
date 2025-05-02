from decimal import Decimal, getcontext
import numpy as np

from slr.temporal_matrix import TemporalMatrix


class RotaryMatrixReader:
    def __init__(self, file: str):
        self.file = file
        self.timestamps = []
        self.memory_map = None
        self.number_of_recordings = -1
        self._load_number_of_recordings()


    def _load_number_of_recordings(self) -> None:
        """
        Reads the number of temporal matrix in the given text file and save it in the corresponding attribute.
        :return:
        """
        with open(self.file, "r") as f:
            for i, line in enumerate(f):
                if i == 5: # number of recordings is written at the 6th line
                    number_of_lines = int(line.strip().split()[0])
                    self.number_of_recordings = number_of_lines
                    break
        print(f"The file {self.file} has {self.number_of_recordings} recordings.")



    def generate_npy_file(self, log: bool = False):
        """
        The rotary matrix file is too large: this method generates a npy file containing all data.
        :return:
        """
        with open(self.file, "r") as rotary_matrices_file:
            current_line = 0
            starting_line = 6
            for line in rotary_matrices_file:

                # Read the number of lines to treat
                # Create an appropriate memory map to store data
                if current_line == 5:
                    number_of_lines = int(line.strip().split()[0])
                    self.memory_map = np.memmap("data/rotary_matrices.npy",
                                                mode='w+',
                                                shape=(number_of_lines, 10),
                                                dtype=np.float64)
                    if log:
                        print(f"'rotary_matrices.npy' file generated ({number_of_lines} lines).")

                # Skip header
                elif current_line >= starting_line:
                    # Remove white blanks and split the string w.r.t the character ' '.
                    data = line.strip().split()

                    # Parse and store each data
                    matrix = list(map(float, data))

                    # Index of the read line from the starting line
                    index = current_line - starting_line

                    # Store the line in the npy file
                    self.memory_map[index] = np.array(matrix)

                    # Log
                    if log:
                        print(f"Line {index}")

                current_line = current_line + 1

        # Save the memory map
        self.memory_map.flush()
        print("Memory map correctly generated!")
        self.memory_map = None


    def read(self, n: int) -> TemporalMatrix:
        """
        Read the n-th temporal matrix from the npy file.
        :param n: integer
        :return: the matrix corresponding to the n-th line in the rotary matrix file
        """
        if self.memory_map is None:
            self.memory_map = np.memmap("data/rotary_matrices.npy", mode="r", shape=(26784001, 10), dtype=np.float64)

        row = self.memory_map[n]

        time = row[0]  # The first value is the time
        matrix = row[1:].reshape((3, 3))  # The next 9 values form a 3x3 matrix
        return TemporalMatrix(time, matrix)
