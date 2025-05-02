from readers.normal_point_reader import NormalPointReader
from readers.rotary_matrix_reader import RotaryMatrixReader
from readers.station_position_reader import StationPositionReader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #np_reader = NormalPointReader("data/normalsPoints_gracefo1_7810_2021-12.txt")
    #np_reader.generate_npy_file(debug=True)
    pos_reader = StationPositionReader("data/stationPosition.7810.slr.txt")
    pos_reader.generate_npy_file(debug=True)
    #matrix_reader = RotaryMatrixReader("data/rotaryMatrix_2021-12.txt")
    #matrix_reader.generate_npy_file(log=True)
    #print(matrix_reader.read(2))
