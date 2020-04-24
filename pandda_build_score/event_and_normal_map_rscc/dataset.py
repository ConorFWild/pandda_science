from typing import NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import gemmi

from torch.utils.data import Dataset, DataLoader

from pandda_types.data import Event


def get_distance(residue,
                 event_centroid,
                 ):
    atom_coords = []
    for atom in residue:
        atom_coords.append([atom.pos[0],
                            atom.pos[1],
                            atom.pos[2],
                            ])

    atom_coords_array = np.array(atom_coords)
    mean_ligand_pos = np.mean(atom_coords_array,
                              axis=0,
                              )

    distance = np.linalg.norm(mean_ligand_pos - event_centroid)
    return distance


def get_receptor_model(path):
    return gemmi.read_structure(path)


def get_ligand_model(path):
    return gemmi.read_structure(path)


def sample_rotation():
    # angles = np.random.rand(3) * np.pi * 2
    # rotation = Rotation.from_euler("xyz",
    #                                [
    #                                    [angles[0], 0, 0],
    #                                    [0, angles[1], 0],
    #                                    [0, 0, angles[2]]
    #                                ],
    #                                )
    # rotation_matrixs = rotation.as_matrix()
    # rotation_matrix = np.matmul(np.matmul(rotation_matrixs[0],
    #                                       rotation_matrixs[1],
    #                                       ),
    #                             rotation_matrixs[2],
    #                             )
    rotation = Rotation.random()
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix


def sample_translation(magnitude=2):
    return (np.random.rand(3) * 2 * magnitude) - magnitude


def sample_map(gemmi_grid,
               centroid,
               rotation,
               translation,
               shape,
               scale=0.25,
               ):
    rotation = np.matmul(rotation, np.eye(3) * scale, )

    offset = -(np.array(shape) / 2) * scale

    rotated_offset = np.matmul(rotation, offset) + translation

    offset_translation = centroid + rotated_offset

    arr = np.zeros(shape,
                   dtype=np.float32,
                   )
    tr = gemmi.Transform()

    tr.mat.fromlist(rotation.tolist())
    tr.vec.fromlist(offset_translation.tolist())
    gemmi_grid.interpolate_values(arr,
                                  tr,
                                  )
    return arr


def get_label(event,
              cutoff=10.0,
              ):
    if event.actually_built:
        if event.distance_to_ligand_model > 0:
            if event.distance_to_ligand_model < cutoff:
                return np.array([0, 1],
                                dtype=np.float32,
                                )
            else:
                return np.array([1, 0],
                                dtype=np.float32,
                                )
        else:
            return np.array([1, 0],
                            dtype=np.float32,
                            )
    else:
        return np.array([1, 0],
                        dtype=np.float32,
                        )


def sample_event_map(event_map,
                     centroid,
                     rotation,
                     translation,
                     shape,
                     ):
    return sample_map(event_map.grid,
                      centroid,
                      rotation,
                      translation,
                      shape,
                      )


def clone_grid(grid):
    new_grid = gemmi.FloatGrid(grid.nu, grid.nv, grid.nw)
    new_grid.spacegroup = grid.spacegroup
    new_grid.set_unit_cell(grid.unit_cell)
    return new_grid


def get_map_from_mtz_path(path):
    mtz = gemmi.read_mtz_file(str(path))
    xmap = mtz.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=6)
    return xmap


class EventAndNormalMapDataset(Dataset):
    def __init__(self, table, rscc_table, shape=np.array([16, 16, 16])):
        self.table: pd.DataFrame = table
        self.rscc_dict = {(row["pandda_name"], row["dtag"], row["event_idx"]): float(row["rscc"])
                          for index, row
                          in rscc_table.iterrows()}
        self.sample_shape = shape

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        event_record = self.table.iloc[item]

        event = Event.from_record(event_record)
        rscc = self.rscc_dict[(event.pandda_name, event.dtag, event.event_idx)]

        # if bool(event.viewed) is False:
        #     sample_dict = {"id": {"pandda_name": event.pandda_name,
        #                           "dtag": event.dtag,
        #                           "event_idx": event.event_idx,
        #                           },
        #                    "data": np.zeros((2,
        #                                      self.sample_shape[0],
        #                                      self.sample_shape[1],
        #                                      self.sample_shape[2],
        #                                      ),
        #                                     dtype=np.float32,
        #                                     ),
        #                    "label": np.array([1, 0], dtype=np.float32),
        #                    "event_map_path": str(event.event_map_path),
        #                    "model_path": str(event.initial_model_path),
        #                    "coords": str([event.x, event.y, event.z]),
        #                    "rscc": 0,
        #                    }
        #
        #     return sample_dict

        if float(rscc) == 0.0:
            sample_dict = {"id": {"pandda_name": event.pandda_name,
                                  "dtag": event.dtag,
                                  "event_idx": event.event_idx,
                                  },
                           "data": np.zeros((2,
                                             self.sample_shape[0],
                                             self.sample_shape[1],
                                             self.sample_shape[2],
                                             ),
                                            dtype=np.float32,
                                            ),
                           "label": np.array([1, 0], dtype=np.float32),
                           "event_map_path": str(event.event_map_path),
                           "model_path": str(event.initial_model_path),
                           "coords": str([event.x, event.y, event.z]),
                           "rscc": 0.0,
                           "rscc_class": np.array([1.0,0.0], dtype=np.float32),
                           }

            return sample_dict

        try:

            event_centroid = np.array([event.x,
                                       event.y,
                                       event.z,
                                       ])

            data_map = get_map_from_mtz_path(event.data_path)
            event_map: gemmi.Grid = gemmi.read_ccp4_map(event.event_map_path)

            rotation = sample_rotation()
            translation = sample_translation()

            data_map_layer = sample_map(data_map,
                                        event_centroid,
                                        rotation,
                                        translation,
                                        self.sample_shape,
                                        )
            event_map_layer = sample_event_map(event_map,
                                               event_centroid,
                                               rotation,
                                               translation,
                                               self.sample_shape,
                                               )
            data = np.stack([data_map_layer, event_map_layer],
                            axis=0,
                            )

            label = get_label(event)

            if rscc <0.7:
                rscc_class = np.array([1.0, 0.0], dtype=np.float32)
            else:
                rscc_class = np.array([0.0,1.0], dtype=np.float32)

            sample_dict = {"id": {"pandda_name": event.pandda_name,
                                  "dtag": event.dtag,
                                  "event_idx": event.event_idx,
                                  },
                           "data": data,
                           "label": label,
                           "event_map_path": str(event.event_map_path),
                           "model_path": str(event.initial_model_path),
                           "coords": str([event.x, event.y, event.z]),
                           "rscc": rscc,
                           "rscc_class": rscc_class,
                           }

            return sample_dict
        except Exception as e:
            # Null result
            # print(e)
            sample_dict = {"id": {"pandda_name": event.pandda_name,
                                  "dtag": event.dtag,
                                  "event_idx": event.event_idx,
                                  # "initial_model_dir": ,
                                  },
                           "data": np.zeros((2,
                                             self.sample_shape[0],
                                             self.sample_shape[1],
                                             self.sample_shape[2],
                                             ),
                                            dtype=np.float32,
                                            ),
                           "label": np.array([1, 0], dtype=np.float32),
                           "event_map_path": str(event.event_map_path),
                           "model_path": str(event.initial_model_path),
                           "coords": str([event.x, event.y, event.z]),
                           "rscc": 0.0,
                           "rscc_class": np.array([1, 0], dtype=np.float32),
                           }

            return sample_dict


def get_dataloader(table,
                   rscc_table,
                   shape,
                   ):
    dataset: Dataset = EventAndNormalMapDataset(table=table,
                                                rscc_table=rscc_table,
                                                shape=shape,
                                                )

    dataloader: DataLoader = DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=20,
                                        )

    return dataloader
