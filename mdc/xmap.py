import numpy as np

from scipy.spatial.transform import Rotation

import gemmi

class PanDDAXMap:
    def __init__(self, grid):
        self.grid = grid

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

        data = np.array(self.grid)

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
        self.grid.spacegroup = gemmi.SpaceGroup(spacegroup)
        unit_cell = state["unit_cell"]
        self.grid.unit_cell = gemmi.UnitCell(unit_cell[0],
                                             unit_cell[1],
                                             unit_cell[2],
                                             unit_cell[3],
                                             unit_cell[4],
                                             unit_cell[5],
                                             )

        for index, val in np.ndenumerate(data):
            self.grid.set_value(index[0], index[1], index[2], data[index])

    @staticmethod
    def from_dataset(dataset, f="FWT", phi="PHWT", sample_rate=2.6):
        grid = dataset.reflections.mtz.transform_f_phi_to_map(f,
                                                  phi,
                                                  sample_rate=sample_rate,
                                                  )
        return PanDDAXMap(grid)

    @staticmethod
    def from_reflections(reflections, f="FWT", phi="PHWT", sample_rate=2.6):
        grid = reflections.mtz.transform_f_phi_to_map(f,
                                                  phi,
                                                  sample_rate=sample_rate,
                                                  )
        return PanDDAXMap(grid)


def sample(xmap, parameters):
    shape = 32
    scale = 0.5

    arr = np.zeros([32, 32, 32], dtype=np.float32)
    tr = gemmi.Transform()
    trans = [parameters[0],
             parameters[1],
             parameters[2],
             ]
    angles = [parameters[3],
              parameters[4],
              parameters[5],
              ]

    rotation_x = Rotation.from_euler("x", angles[0])
    rotation_y = Rotation.from_euler("y", angles[1])
    rotation_z = Rotation.from_euler("z", angles[2])

    scale = np.eye(3)*0.5

    rotation = np.matmul(rotation_x.as_matrix(),
                         np.matmul(rotation_y.as_matrix(),
                                   rotation_z.as_matrix(),
                                   ),
                         )
    rotation = np.matmul(rotation, scale)

    trans = trans - np.matmul(rotation, np.array([shape, shape, shape])/2)

    tr.mat.fromlist(rotation.tolist())
    tr.vec.fromlist(trans)
    xmap.grid.interpolate_values(arr, tr)
    return arr