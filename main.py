from readers.normal_point_reader import NormalPointReader
from readers.rotary_matrix_reader import RotaryMatrixReader
from readers.station_position_reader import StationPositionReader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Normal points
    np_reader = NormalPointReader("data/normals_points_file.txt")
    np_reader.generate_npy_file(debug=True)

    # Station positions
    pos_reader = StationPositionReader("data/station_position.txt")
    pos_reader.generate_npy_file(debug=True)

    # Rotary matrices
    matrix_reader = RotaryMatrixReader("data/rotary_matrix_file.txt")
    matrix_reader.generate_npy_file(debug=True)
