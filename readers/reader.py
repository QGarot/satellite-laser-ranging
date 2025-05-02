import numpy as np


class Reader:
    def __init__(self, file: str):
        self.file = file
        self.memory_map = None
        self.number_of_recordings = -1
        self.number_of_columns = -1
        self.npy_file = self._get_npy_file_name()
        self._load_number_of_recordings()
        self._load_number_of_columns()

    def _get_npy_file_name(self) -> str:
        """
        Generates the string of the npy file name.
        :return: complete npy file name
        """
        base = self.file.split("/")[1].split(".")[0]
        return f"data/binary/{base}.npy"

    def _load_number_of_columns(self) -> None:
        """
        Reads the number of columns from the text file.
        """
        with open(self.file, "r") as f:
            for i, line in enumerate(f):
                if i == 6: # starting line
                    self.number_of_columns = len(line.strip().split())
                    break
        print(f"The file {self.file} has {self.number_of_columns} columns.")

    def _load_number_of_recordings(self) -> None:
        """
        Reads the number of recordings from the text file.
        """
        with open(self.file, "r") as f:
            for i, line in enumerate(f):
                if i == 5: # number of recordings is written at the 6th line
                    number_of_lines = int(line.strip().split()[0])
                    self.number_of_recordings = number_of_lines
                    break
        print(f"The file {self.file} has {self.number_of_recordings} recordings.")

    def generate_npy_file(self, debug: bool = False) -> None:
        """
        Generates a npy file containing all data of the given file.
        :param debug: true for showing log messages, false otherwise.
        """
        if debug:
            print(f"{self.file}: generating npy file with "
                  f"{self.number_of_recordings} rows and "
                  f"{self.number_of_columns} columns...")

        self.memory_map = np.memmap(self.npy_file,
                                    mode="w+",
                                    shape=(self.number_of_recordings, self.number_of_columns),
                                    dtype=float)

        starting_line = 6
        with open(self.file, "r") as f:
            for i, line in enumerate(f):
                # Skip the header
                if i >= starting_line:
                    # Remove unnecessary whitespaces and split the string w.r.t the whitespace character.
                    data = line.strip().split()

                    # Parse data
                    parsed_data = list(map(float, data))

                    # Index of the current line from the starting line
                    index = i - starting_line

                    # Store parsed data into the memory map
                    self.memory_map[index] = np.array(parsed_data)

                    if debug:
                        print(f"{self.file}: line {index} saved.")

        # Save the memory map on the binary folder
        self.memory_map.flush()
        if debug:
            print(f"{self.file}: npy file generated.")
        self.memory_map = None

    def read(self, n: int) -> np.ndarray[float]:
        """
        Reads the data array of the index n from the npy file.
        :param n: index
        :return: 'memory_map[n]'
        """
        if self.memory_map is None:
            self.memory_map = np.memmap(self.npy_file,
                                        mode="r",
                                        shape=(self.number_of_recordings, self.number_of_columns),
                                        dtype=float)

        return self.memory_map[n]
