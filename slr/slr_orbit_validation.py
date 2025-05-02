from readers.normal_point_reader import NormalPointReader
from readers.rotary_matrix_reader import RotaryMatrixReader
from readers.station_position_reader import StationPositionReader
from math import floor
import numpy as np


class SLROrbitValidation:
    def __init__(self):
        # Normal points
        self.np_reader = NormalPointReader("data/normals_points_file.txt")

        # Station positions
        self.pos_reader = StationPositionReader("data/station_position.txt")

        # Rotary matrices
        self.matrix_reader = RotaryMatrixReader("data/rotary_matrix_file.txt")
        #self.matrix_reader.generate_npy_file(debug=True)
        print(self.matrix_reader.get_temporal_rotary_matrix(2))

    def nearest_neighbour(self, t: float, k: int, debug: bool = False) -> list:
        """
        Finds the value t0 that is closest to t such that the binary file contains a rotation matrix with the timestamp
        t0, denoted R(t0).

        Returns the list containing:
            - the k matrices stored just before R(t0)
            - R(t0)
            - the k matrices stored just after R(t0)
        These matrices are called "neighbour matrices of t".

        :param debug: print the logs or not
        :param k: size of the neighbourhood
        :param t: timestamp for which we want to find neighbour matrices
        :return:  neighbourhood of t with size k
        """
        d = lambda tho: abs(tho - t)

        # Initialization
        # Index for which t0 is reached
        index = 0
        # R(t0)
        matrix = self.matrix_reader.get_temporal_rotary_matrix(0)
        # Nearest timestamp t0
        t0 = matrix.time
        # Distance between t and t0
        d_min = d(t0)


        # TODO: write the doc
        mem = self.matrix_reader.memory_map[:, 0]
        t_tilde = floor(t * 1000) / 1000
        p = np.where(mem == t_tilde)[0][0]
        for i in range(p, self.matrix_reader.number_of_recordings):
            temp_matrix = self.matrix_reader.get_temporal_rotary_matrix(i)
            temp_t0 = temp_matrix.time
            distance = d(temp_t0)
            if distance < d_min:
                d_min = distance
                index = i
                matrix = temp_matrix
                t0 = temp_t0
                if debug:
                    print(f"Temporary minimum (distance): d = {d_min}")
            else:
                break

        if debug:
            print(f"Index of R(t0): {index}. |t0 - t| = {d_min}\n"
                  f"{matrix}")

        predecessors = [self.matrix_reader.get_temporal_rotary_matrix(index - m) for m in range(1, k + 1)]
        successors = [self.matrix_reader.get_temporal_rotary_matrix(index + m) for m in range(1, k + 1)]
        neighbourhood = predecessors + [matrix] + successors

        return neighbourhood
