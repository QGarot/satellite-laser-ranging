from readers.reader import Reader


class ReaderTest:
    @staticmethod
    def test_npy_file_name(txt_file: str) -> None:
        reader = Reader(txt_file)
        print(reader.npy_file)

    @staticmethod
    def txt2npy(txt_file: str) -> None:
        reader = Reader(txt_file)
        reader.generate_npy_file_v2()

        print(reader.loaded_data[0])


ReaderTest.txt2npy("../statictest/txt/np/normalsPoints_gracefo1_7810_2021-12.txt")