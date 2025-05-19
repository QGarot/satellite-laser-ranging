from matplotlib import pyplot as plt

from include.interpolation import neville
from readers.normal_point_reader import NormalPointReader
from readers.orbit_reader import OrbitReader
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
from config import config


class SLROrbitValidation:
    def __init__(self):
        # Normal points
        self.np_reader = NormalPointReader(f"{config.DATA_DIRECTORY}/txt/np/{config.NORMAL_POINT}")

        # Station positions
        self.pos_reader = StationPositionReader(f"{config.DATA_DIRECTORY}/txt/positions/{config.STATION_POSITIONS}")

        # Rotary matrices
        self.matrix_reader = RotaryMatrixReader(f"{config.DATA_DIRECTORY}/txt/matrices/{config.MATRICES}")

        # Satellite positions
        # List of orbit readers
        self.orbits = []
        self.load_satellite_orbits()

    @staticmethod
    def display_3d_vectors(vectors: np.ndarray, additional_vectors: np.ndarray, interactive_mode: bool = False) -> None:
        """
        TODO: write the doc
        :param vectors:
        :param additional_vectors:
        :param interactive_mode:
        :return:
        """
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

    def first_orbit_validation(self, debug: False) -> None:
        """
        For each normal point, compares the measured distance and the computed distance.
        The 'computed distance' is the euclidian norm of the difference between:
            - the position of the satellite at t_bounce
            - the position of the ground station at t_emission
        both written in CRF (Celestial Reference Frame).
        :return:
        """
        measured_distances = []
        computed_distances = []
        transmit_timestamps = []
        residuals = []

        for k in range(0, self.np_reader.number_of_recordings):
            if debug:
                print(f"\n\n------------------------------ Normal point {k} ------------------------------")

            normal_point = self.np_reader.get_normal_points(k)
            station_pos = self.get_crf_position(normal_point.time, 1, visualization=False, debug=debug)
            satellite_pos = self.get_satellite_position(normal_point.get_time_bounce(), 1, visualization=False, debug=debug)
            dist = np.linalg.norm(satellite_pos.vector - station_pos.vector)
            residual = normal_point.measured_range - dist

            transmit_timestamps.append(normal_point.time)
            measured_distances.append(normal_point.measured_range)
            computed_distances.append(dist)
            residuals.append(residual)

            if debug:
                print(f"=====> NP timestamp [{normal_point.time:.10e}] | "
                      f"t_bounce [{normal_point.get_time_bounce():.6e}] | "
                      f"measured dist. [{normal_point.measured_range:.6e}] | "
                      f"computed dist. [{dist:.6e}] | "
                      f"error [{residual:.6e}]")


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

    @staticmethod
    def interpolation(temporal_positions: list[TemporalPosition], t: float) -> tuple[float, float, float]:
        """
        Performs an interpolation using Neville-Aitken's algorithm.
        :param t: timestamp for which we want to evaluate the interpolation
        :param temporal_positions: list of temporal positions
        :return: interpolated vector (x,y,z) at the given timestamp
        """
        # x interpolation
        data_set_x = [(p.time, p.x) for p in temporal_positions]
        x = neville(data_set_x, t)

        # y interpolation
        data_set_y = [(p.time, p.y) for p in temporal_positions]
        y = neville(data_set_y, t)

        # z interpolation
        data_set_z = [(p.time, p.z) for p in temporal_positions]
        z = neville(data_set_z, t)

        return x, y, z

    def nearest_neighbour(self, t: float, k: int) -> list[TemporalMatrix]:
        """
        Finds the value t0 that is closest to t such that the binary file contains a rotation matrix with the timestamp
        t0, denoted R(t0).

        Returns the list containing:
            - the k matrices stored just before R(t0)
            - R(t0)
            - the k matrices stored just after R(t0)
        These matrices are called "neighbour matrices of t".

        :param k: size of the neighbourhood
        :param t: timestamp for which we want to find neighbour matrices
        :return: neighbourhood of t with size k
        """
        n = self.matrix_reader.get_index(t)

        predecessors = [self.matrix_reader.get_temporal_rotary_matrix(n - m) for m in range(1, k + 1)][::-1] # reverse order
        successors = [self.matrix_reader.get_temporal_rotary_matrix(n + m) for m in range(1, k + 1)]
        neighbourhood = predecessors + [self.matrix_reader.get_temporal_rotary_matrix(n)] + successors

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

    def get_crf_position(self, t: float, order: int, visualization: bool, debug: bool = False) -> TemporalPosition:
        """
        Returns the temporal position instance corresponding to the position (in the CRF) that the station would have at the given timestamp.

        The algorithm is the following:
        (1) Get the position of the station (in the TRF) for the corresponding day (station positions npy file gives the
        station's position for each day)
        (2) Get the neighbour matrices of t w.r.t the given order (size of the neighbourhood)
        (3) For each matrice in the defined neighbourhood, compute the temporal position of the station (in the CRF) and store the
        result in a list (cf 'crf_points')
        (4) At this stage, we have a list of temporal positions in the CRF from which we want to interpolate the position
        of the station at the given timestamp. We perform an interpolation using Neville-Aitken's algorithm for each component.
        (5) Return the temporal position object corresponding to the constructed vector and the given timestamp.
        :param debug:
        :param visualization:
        :param t: timestamp for which we want to find the position of the station
        :param order: order of the interpolation
        :return: temporal position instance corresponding to the position (in the CRF) of the station at the given timestamp
        """
        if debug:
            print(f"=====> [Ground station] 'get_crf_position()':")

        index = self.pos_reader.get_index(t)
        trf_pos = self.pos_reader.get_temporal_position(index)

        if debug:
            print(f"==> TRF pos: {trf_pos}")

        nearest_neighbour = self.nearest_neighbour(t, order)

        # List of CRF temporal positions corresponding to the nearest matrices
        crf_points = []

        if debug:
            print(f"==> Matrices neighbourhood:")
        for rotation in nearest_neighbour:
            # Get the matrix and its corresponding timestamp
            rotation_matrix = rotation.matrix
            timestamp = rotation.time
            if debug:
                print(f"R({timestamp})=\n{rotation_matrix}")
            # Conversion TRF -> CRF
            crf_vector = self.trf_to_crf(rotation_matrix, trf_pos.vector)
            # Create the temporal point associated and save it
            crf_pos = TemporalPosition(timestamp, crf_vector[0], crf_vector[1], crf_vector[2])
            crf_points.append(crf_pos)

        x, y, z = self.interpolation(crf_points, t)
        res = TemporalPosition(t, x, y, z)

        if debug:
            print(f"==> Interpolated CRF pos:\n{res}")

        # 3D visualisation
        if visualization:
            self.display_3d_vectors(np.array([crf_point.vector for crf_point in crf_points]),
                                    np.array([res.vector]),
                                    False)

        return res

    def load_satellite_orbits(self) -> None:
        """
        For each text orbit file, a NPY file is created through a ReducedDynamicOrbitReader object, which is saved into
        the 'self.orbits' attribute.
        """
        folder = Path(f"{config.DATA_DIRECTORY}/txt/orbits")
        files = [str(f) for f in folder.iterdir() if f.is_file()]
        for txt_file in files:
            reader = OrbitReader(txt_file)
            self.orbits.append(reader)


    def get_satellite_position(self, t: float, order: int, visualization: bool, debug: bool = False) -> TemporalPosition:
        """
        Returns the temporal position instance corresponding to the position that the satellite would have at the given timestamp.
        The algorithm is the following:

        (1) Search the position in the right file
            * Knowing 't', we can determine its corresponding day, denoted 'day'.
            * Knowing 'day', we select the appropriate file where the position needs to be searched.
        (2) Look for the nearest timestamp (denoted t0) of 't' such that there exists a recorded position for 't0'.
        (3) Determine the neighbour positions.
        (4) Perform an interpolation using Neville-Aitken's algorithm for each component of the neighbour positions.
        (5) Return the temporal position object corresponding to the constructed vector and the given timestamp.
        :param debug:
        :param visualization:
        :param order:
        :param t:
        :return:station
        """
        if debug:
            print(f"=====> 'get_satellite_position()':")

        # Day corresponding to the given time t
        day = floor(t)

        # Reader
        reduced_dynamic_orbit_reader: OrbitReader | None = None

        for reader in self.orbits:
            file_starting_time = floor(reader.t0)
            # If we find the orbit file corresponding to 'day', then we save it and break the loop.
            if day == file_starting_time:
                reduced_dynamic_orbit_reader = reader
                break

        # We look for the nearest t0 such that there is a recorded position:
        n = reduced_dynamic_orbit_reader.get_index(t)

        predecessors = [reduced_dynamic_orbit_reader.get_temporal_position(n - m) for m in range(1, order + 1)][::-1]
        successors = [reduced_dynamic_orbit_reader.get_temporal_position(n + m) for m in range(1, order + 1)]
        neighbourhood = predecessors + [reduced_dynamic_orbit_reader.get_temporal_position(n)] + successors

        x, y, z = self.interpolation(neighbourhood, t)

        res = TemporalPosition(t, x, y, z)

        if debug:
            print(f"==> Interpolated pos:\n{res}")

        # 3D visualisation
        if visualization:
            self.display_3d_vectors(np.array([neighbour.vector for neighbour in neighbourhood]), np.array([res.vector]), False)

        return res
