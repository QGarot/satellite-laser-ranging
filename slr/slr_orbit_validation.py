from matplotlib import pyplot as plt

from include.interpolation import neville
from readers.normal_point_reader import NormalPointReader
from readers.reduced_dynamic_orbit_reader import ReducedDynamicOrbitReader
from readers.rotary_matrix_reader import RotaryMatrixReader
from readers.station_position_reader import StationPositionReader
from math import floor
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import webbrowser
import plotly.io as pio

from slr.temporal_matrix import TemporalMatrix
from slr.temporal_position import TemporalPosition


class SLROrbitValidation:
    def __init__(self):
        # Normal points
        self.np_reader = NormalPointReader("static/txt/slr/normals_points_file.txt")

        # Station positions
        self.pos_reader = StationPositionReader("static/txt/slr/station_position.txt")

        # Rotary matrices
        self.matrix_reader = RotaryMatrixReader("static/txt/slr/rotary_matrix_file.txt")

        # Satellite positions
        # List of orbit readers
        self.orbits = []
        self.load_satellite_orbits(False)

    @staticmethod
    def display_3d_vectors(vectors: np.ndarray, additional_vectors: np.ndarray, interactive_mode: bool = False) -> None:
        if interactive_mode:
            pio.renderers.default = 'browser'
            x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

            fig = go.Figure(data=[
                go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='red'),
                             line=dict(color='blue')),
            ])

            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                title='3D interactive visualization',
            )

            fig.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for v in vectors:
                ax.quiver(0, 0, 0, v[0], v[1], v[2], color='b', arrow_length_ratio=0.1)

            for additional_v in additional_vectors:
                ax.quiver(0, 0, 0, additional_v[0], additional_v[1], additional_v[2], color='r', arrow_length_ratio=0.1)

            ax.set_xlim([0, np.max(vectors[:, 0]) + 1])
            ax.set_ylim([0, np.max(vectors[:, 1]) + 1])
            ax.set_zlim([0, np.max(vectors[:, 2]) + 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D visualisation')

            plt.show()

    def first_orbit_validation(self) -> None:
        """
        TODO: write the doc
        :return:
        """
        measured_distances = []
        computed_distances = []
        transmit_timestamps = []
        residuals = []

        for k in range(0, self.np_reader.number_of_recordings):
            normal_point = self.np_reader.get_normal_points(k)
            station_pos = self.get_crf_position(normal_point.time, 3)
            satellite_pos = self.get_satellite_position(normal_point.get_time_bounce(), 3)
            dist = np.linalg.norm(satellite_pos.vector - station_pos.vector)
            residual = normal_point.measured_range - dist

            transmit_timestamps.append(normal_point.time)
            measured_distances.append(normal_point.measured_range)
            computed_distances.append(dist)
            residuals.append(residual)

            print(f"NP timestamp [{normal_point.time:.10e}], "
                  f"t_bounce [{normal_point.get_time_bounce():.10e}], "
                  f"measured dist. [{normal_point.measured_range:.2e}], "
                  f"computed dist. [{dist:.2e}],"
                  f"error [{residual:.2e}]")


        x = np.array(transmit_timestamps)
        y1 = np.array(computed_distances)
        y2 = np.array(measured_distances)
        y3 = np.array(residuals)

        #plt.plot(x, y1, "o-", color="green", label="Measured distances")
        #plt.plot(x, y2, "o--", color="red", label="Computed distances")
        plt.plot(x, y3, "o--", color="#ff7f0e", label="Residuals")

        plt.title("Residuals vs. laser transmission time")
        plt.xlabel("Laser transmission timestamps [MJD]")
        plt.ylabel("Distance [m]")
        plt.legend()
        plt.grid(True)
        plt.show()

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
        return rotation_matrix.T @ x

    def get_station_position(self, t: float) -> TemporalPosition:
        """
        Returns the temporal position of the station corresponding to given timestamp t.
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

    def get_crf_position(self, t: float, order: int, visualization: bool = False) -> TemporalPosition:
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
        :param visualization:
        :param t: timestamp for which we want to find the position of the station
        :param order: order of the interpolation
        :return: temporal position instance corresponding to the position (in the CRF) of the station at the given timestamp
        """
        trf_pos = self.get_station_position(t)
        nearest_neighbour = self.nearest_neighbour(t, order)
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
        data_set = [(temporal_crf_point.time, temporal_crf_point.x) for temporal_crf_point in crf_points]
        x = neville(data_set, t)

        # y interpolation
        data_set = [(temporal_crf_point.time, temporal_crf_point.y) for temporal_crf_point in crf_points]
        y = neville(data_set, t)

        # z interpolation
        data_set = [(temporal_crf_point.time, temporal_crf_point.z) for temporal_crf_point in crf_points]
        z = neville(data_set, t)

        res = TemporalPosition(t, x, y, z)

        # 3D visualisation
        if visualization:
            self.display_3d_vectors(np.array([crf_point.vector for crf_point in crf_points]),
                                    np.array([res.vector]),
                                    False)

        return res

    def load_satellite_orbits(self, npy_creation: bool) -> None:
        """
        For each text orbit file, a NPY file is created through a ReducedDynamicOrbitReader object, which is saved into
        the 'self.orbits' attribute.

        If the parameter 'npy_creation' is set to false, that means npy files are already created. Then the methods just
        save the orbit readers.
        :param npy_creation: true if the npy files have to be generated.
        """
        folder = Path(f"static/txt/orbits")
        files = [str(f) for f in folder.iterdir() if f.is_file()]
        for txt_file in files:
            reader = ReducedDynamicOrbitReader(txt_file)
            if npy_creation:
                reader.generate_npy_file()
            self.orbits.append(reader)


    def get_satellite_position(self, t: float, order: int, visualization: bool = False) -> TemporalPosition:
        """
        Returns a temporal position instance of the satellite corresponding to its position at the given timestamp.
        The algorithm is the following:

        (1) Search the position in the right file
            * Knowing 't', we can determine its corresponding day, denoted 'day'.
            * Knowing 'day', we select the appropriate file where the position needs to be searched.
        (2) Look for the nearest timestamp (denoted t0) of 't' such that there exists a recorded position for 't0'.
        (3) Determine the neighbour positions.
        (4) Perform an interpolation using Neville-Aitken's algorithm for each component of the neighbour positions.
        (5) Return the temporal position object corresponding to the constructed vector and the given timestamp.

        :param visualization:
        :param order:
        :param t:
        :return:station
        """
        # Define what is the distance between an arbitrary timestamp tho and t
        d = lambda tho: abs(tho - t)

        # Day corresponding to the given time t
        day = floor(t)

        # Reader
        reduced_dynamic_orbit_reader = None

        for reader in self.orbits:
            file_starting_time = reader.memory_map[0][0]
            # If we find the orbit file corresponding to 'day', then we save it and break the loop.
            if day == file_starting_time:
                reduced_dynamic_orbit_reader = reader
                break

        # We look for the nearest t0 such that there is a recorded position:

        # All timestamps recorded in the considered file
        timestamps = reduced_dynamic_orbit_reader.memory_map[:, 0]

        # Initialization
        k = 0
        timestamp = timestamps[k]
        d_min = d(timestamp)

        for i in range(k + 1, len(timestamps)):
            current_timestamp = timestamps[i]
            delta = d(current_timestamp)
            if delta < d_min:
                d_min = delta
                timestamp = current_timestamp
                k = i

        # k is the index of the position we are looking for
        pos = reduced_dynamic_orbit_reader.get_temporal_position(k)

        predecessors = [reduced_dynamic_orbit_reader.get_temporal_position(k - m) for m in range(1, order + 1)][::-1]
        successors = [reduced_dynamic_orbit_reader.get_temporal_position(k + m) for m in range(1, order + 1)]
        neighbourhood = predecessors + [pos] + successors

        # x interpolation
        data_set = [(point.time, point.x) for point in neighbourhood]
        x = neville(data_set, t)

        # y interpolation
        data_set = [(point.time, point.y) for point in neighbourhood]
        y = neville(data_set, t)

        # z interpolation
        data_set = [(point.time, point.z) for point in neighbourhood]
        z = neville(data_set, t)

        res = TemporalPosition(t, x, y, z)

        # 3D visualisation
        if visualization:
            self.display_3d_vectors(np.array([neighbour.vector for neighbour in neighbourhood]), np.array([res.vector]), False)

        return res
