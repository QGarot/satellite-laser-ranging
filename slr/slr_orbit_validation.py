from include.interpolation import neville
from readers.normal_point_reader import NormalPointReader
from readers.rotary_matrix_reader import RotaryMatrixReader
from readers.station_position_reader import StationPositionReader
from math import floor
import numpy as np

from slr.temporal_matrix import TemporalMatrix
from slr.temporal_position import TemporalPosition


class SLROrbitValidation:
    def __init__(self):
        # Normal points
        self.np_reader = NormalPointReader("data/normals_points_file.txt")

        # Station positions
        self.pos_reader = StationPositionReader("data/station_position.txt")

        # Rotary matrices
        self.matrix_reader = RotaryMatrixReader("data/rotary_matrix_file.txt")

    def nearest_neighbour(self, t: float, k: int, debug: bool = False) -> list[TemporalMatrix]:
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
        :return: neighbourhood of t with size k
        """
        # Define what is the distance between an arbitrary timestamp tho and t
        d = lambda tho: abs(tho - t)

        # /!\ Important observation /!\
        # Let's call t_min and t_max the smallest and the largest timestamps recorded in the file.
        # (*) In the npy file, all timestamps in [t_min, t_max] with exactly three digits after the decimal point are recorded.
        # That allows to find an initial approximation of t from which we will look for t0, without scanning the array from the beginning.

        # We create the (sorted) array of all timestamps in the npy file.
        timestamps = self.matrix_reader.memory_map[:, 0]
        # Given a real timestamp t, we compute its rounded down with three digits after the decimal point.
        t0 = floor(t * 1000) / 1000
        # We look for the index p such that mem[p] = t0, which is guaranteed to exist due to (*).
        p = np.where(timestamps == t0)[0][0]
        # Distance between t and t0
        d_min = d(t0)

        if debug:
            print(f"[row = {p}], t = {t0}, d_min = {d_min}")

        for i in range(p + 1, self.matrix_reader.number_of_recordings):
            temp_t0 = timestamps[i]
            temp_distance = d(temp_t0)

            if debug:
                print(f"[row = {i}], t = {temp_t0}, d_min = {temp_distance}")

            if temp_distance < d_min: # if we find a better approximation of t, then we save it as following.
                d_min = temp_distance
                p = i
                t0 = temp_t0
            else: # Otherwise, as 'timestamps' is sorted, we cannot find a better approximation of t anymore.
                if debug:
                    print(f"The value t = {t} has been exceeded")
                break

        temporal_matrix = self.matrix_reader.get_temporal_rotary_matrix(p)
        if debug:
            print(f"Index of R(t0): {p}. |t0 - t| = {d_min}\n"
                  f"{temporal_matrix.matrix}")

        predecessors = [self.matrix_reader.get_temporal_rotary_matrix(p - m) for m in range(1, k + 1)][::-1] # reverse order
        successors = [self.matrix_reader.get_temporal_rotary_matrix(p + m) for m in range(1, k + 1)]
        neighbourhood = predecessors + [temporal_matrix] + successors

        return neighbourhood

    @staticmethod
    def trf_to_crf(rotation_matrix: np.ndarray, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Returns the corresponding vector in Celestial Reference Frame (CRF).
        :param rotation_matrix: rotation matrix
        :param x: vector given in Terrestrial Reference Frame (TRF)
        :return: x rotated
        """
        return rotation_matrix @ x

    def get_station_position(self, t: float) -> TemporalPosition:
        """
        Returns the temporal position corresponding to given timestamp t.
        :param t: timestamp for which we want to find neighbour matrices
        :return: temporal position corresponding to t
        """
        # We create the (sorted) array of all timestamps in the npy file.
        timestamps = self.pos_reader.memory_map[:, 0]
        # Get the corresponding day
        t0 = float(floor(t)) + 0.5
        # Find the index p such that timestamps[p] = t0
        p = np.where(timestamps == t0)[0][0]

        return self.pos_reader.get_temporal_position(p)

    def get_crf_position(self, t: float, order: int) -> TemporalPosition:
        """
        Returns a temporal position instance corresponding to the position (in the CRF) of the station at the given timestamp.

        The algorithm is the following:
        (1) Get the position of the station (in the TRF) for the corresponding day (station positions npy file gives the
        station's position for each day)
        (2) Get the neighbour matrices of t w.r.t the given order (size of the neighbourhood)
        (3) For each matrice in the defined neighbourhood, compute the temporal position of the station (in the CRF) and store the
        result in a list (cf 'crf_points')
        (4) At this stage, we have a list of temporal positions in the CRF from which we want to interpolate the position
        of the station at the given timestamp. We perform an interpolation using Neville-Aitken's algorithm for each component.
        (5) Return the temporal position object corresponding to the constructed vector and the given timestamp.
        :param t: timestamp for which we want to find the position of the station
        :param order: order of the interpolation
        :return: temporal position instance corresponding to the position (in the CRF) of the station at the given timestamp
        """
        trf_pos = self.get_station_position(t)
        nearest_neighbour = self.nearest_neighbour(t, order)
        print(type(nearest_neighbour[0]))
        # List of CRF temporal positions corresponding to the nearest matrices
        crf_points = []

        for rotation in nearest_neighbour:
            # Get the matrix and its corresponding timestamp
            rotation_matrix = rotation.matrix
            timestamp = rotation.time
            # Conversion TRF -> CRF
            crf_vector = self.trf_to_crf(rotation_matrix, trf_pos.vector)
            # Create the temporal point associated and save it
            crf_pos = TemporalPosition(timestamp, crf_vector[0], crf_vector[1], crf_vector[2])
            crf_points.append(crf_pos)

        # x interpolation
        data_set = [(temporal_crf_point.time, temporal_crf_point.vector[0]) for temporal_crf_point in crf_points]
        x = neville(data_set, t)

        # y interpolation
        data_set = [(temporal_crf_point.time, temporal_crf_point.vector[1]) for temporal_crf_point in crf_points]
        y = neville(data_set, t)

        # z interpolation
        data_set = [(temporal_crf_point.time, temporal_crf_point.vector[2]) for temporal_crf_point in crf_points]
        z = neville(data_set, t)

        return TemporalPosition(t, x, y, z)
