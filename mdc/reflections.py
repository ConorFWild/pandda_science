import gemmi
import numpy


class PanDDAReflections:
    def __init__(self, mtz):
        self.mtz = mtz

    @staticmethod
    def from_file(path):
        mtz = gemmi.read_mtz_file(str(path))
        return PanDDAReflections(mtz)

    def truncate_reflections(self, resolution):
        all_data = numpy.array(self.mtz, copy=False)
        self.mtz.set_data(all_data[self.mtz.make_d_array() >= resolution])

    def get_resolution_high(self):
        return self.mtz.resolution_high()


