class NormalPoint:
    def __init__(self, time: float, measured_range: float, accuracy: float, redundancy: float, window: float, wavelength: float):
        self.time = time
        self.measured_range = measured_range
        self.accuracy = accuracy
        self.redundancy = redundancy
        self.window = window
        self.wavelength = wavelength

    def __str__(self) -> str:
        return (f"time: {self.time}, "
                f"range: {self.measured_range:.18e}, "
                f"accuracy: {self.accuracy:.18e}, "
                f"redundancy: {self.redundancy:.18e}, "
                f"window: {self.window:.18e}, "
                f"wavelength: {self.wavelength:.18e}")