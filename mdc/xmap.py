import numpy as np

from scipy.spatial.transform import Rotation

import gemmi

class PanDDAXMap:
    def __init__(self, grid):
        self.grid = grid

    @staticmethod
    def from_dataset(dataset, f="FWT", phi="PHWT", sample_rate=2.6):
        grid = dataset.mtz.transform_f_phi_to_map(f,
                                                  phi,
                                                  sample_rate=sample_rate,
                                                  )
        return PanDDAXMap(grid)


def sample(xmap, parameters):
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

    rotation = np.matmul(rotation_x.as_matrix(),
                         np.matmul(rotation_y.as_matrix(),
                                   rotation_z.as_matrix(),
                                   ),
                         )

    tr.mat.fromlist(rotation.tolist())
    tr.vec.fromlist(trans)
    xmap.grid.interpolate_values(arr, tr)
    return arr