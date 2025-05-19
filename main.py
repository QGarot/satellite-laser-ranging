from readers.normal_point_reader import NormalPointReader
from readers.reader import Reader
from readers.rotary_matrix_reader import RotaryMatrixReader
from slr.slr_orbit_validation import SLROrbitValidation

# Tests
def test_txt2npy():
    r = RotaryMatrixReader("statictest/txt/matrices/rotaryMatrix_2021-12.txt")
    #print(r.get_temporal_rotary_matrix_by_time(59549.00033564814924000))

if __name__ == '__main__':
    orbit_validation = SLROrbitValidation()
    # Residual (difference between the measured range and the real distance) vs. transmission time
    orbit_validation.first_orbit_validation(debug=True)

    # test_txt2npy()
