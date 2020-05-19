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

    def __getstate__(self):
        spacegroup = self.grid.spacegroup.number

        unit_cell_gemmi = self.grid.unit_cell
        unit_cell = (unit_cell_gemmi.a,
                     unit_cell_gemmi.b,
                     unit_cell_gemmi.c,
                     unit_cell_gemmi.alpha,
                     unit_cell_gemmi.beta,
                     unit_cell_gemmi.gamma,
                     )

        data = numpy.array(self.grid)

        state = {"spacegroup": spacegroup,
                 "unit_cell": unit_cell,
                 "data": data,
                 }

        return state

    def __setstate__(self, state):
        data = state["data"]
        self.grid = gemmi.FloatGrid(data.shape[0],
                                    data.shape[1],
                                    data.shape[2],
                                    )
        spacegroup = state["spacegroup"]
        unit_cell = state["unit_cell"]

        mtz = gemmi.Mtz()
        mtz.spacegroup = gemmi.find_spacegroup_by_number(spacegroup)
        mtz.cell.set(unit_cell[0],
                                             unit_cell[1],
                                             unit_cell[2],
                                             unit_cell[3],
                                             unit_cell[4],
                                             unit_cell[5],)


        for dataset_name, dataset in datasets.items():
            mtz.add_dataset(dataset_name)
            for column_name, column in dataset.items():
                mtz.add_column(column_name, column)

        mtz.set_data(data)



