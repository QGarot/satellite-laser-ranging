class NormalPoint:
    def __init__(self, time: float, measured_range: float, accuracy: float, redundancy: float, window: float, wavelength: float):
        self.time = time
        self.measured_range = measured_range
        self.accuracy = accuracy
        self.redundancy = redundancy
        self.window = window
        self.wavelength = wavelength

    def get_time_bounce(self):
        """
        Gets the timestamp when the laser light is bounced back on the retro reflector. According to the range value
        defined in https://github.com/groops-devs/groops/blob/main/source/programs/conversion/slr/crd2NormalPoints.cpp,
        this method returns t_transmit + 0.5 * time_of_flight.
        :return: t_bounce, timestamp when the laser light is bounced back on the retro reflector.
        """
        return self.time + (self.measured_range / 3.00e8)

    def __str__(self) -> str:
        return (f"time: {self.time}, "
                f"range: {self.measured_range:.18e}, "
                f"accuracy: {self.accuracy:.18e}, "
                f"redundancy: {self.redundancy:.18e}, "
                f"window: {self.window:.18e}, "
                f"wavelength: {self.wavelength:.18e}")