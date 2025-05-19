import numpy as np
from os.path import exists


class Reader:
    def __init__(self, file: str):
        self.file = file
        self.npy_file = self._get_npy_file_name()

        if exists(self.npy_file):
            print(f"{self.npy_file} already exists.")
        else:
            print(f"{self.npy_file}: generation...")
            self.generate_npy_file_v2()

        self.loaded_data = np.load(self.npy_file)
        self.number_of_recordings = self.loaded_data[0]
        self.number_of_columns = self.loaded_data[1]

    def _get_npy_file_name(self) -> str:
        """
        Generates the string of the npy file name.
        :return: complete npy file name
        """
        base = self.file.split("/")[-1].split(".")[0]
        # Now we look for the folder (in the txt directory) containing the current file 'self.file'
        folder = self.file.split("/")[-2]
        return f"statictest/binary/{folder}/{base}.npy"

    def generate_npy_file_v2(self) -> None:
        """
        Generates a npy file containing all data of the given file.
        """
        try:
            data = np.loadtxt(self.file, skiprows=6)
            np.save(self.npy_file, data)
        except Exception as e:
            print(f"{self.npy_file}: cannot load this file.\n{e}")

    def read(self, n: int) -> np.ndarray[float]:
        """
        Reads the data array of the index n from the npy file.
        :param n: index
        :return: 'memory_map[n]'
        """
        return self.loaded_data[n]
