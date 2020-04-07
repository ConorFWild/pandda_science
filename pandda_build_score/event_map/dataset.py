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


def sample_translation(magnitude=3):
    return (np.random.rand(3) * 2 * magnitude) - magnitude


def sample_map(gemmi_grid,
               centroid,
               rotation,
               translation,
               shape,
               ):
    offset = translation - (np.array(shape) / 2)

    rotated_offset = np.matmul(rotation, offset)

    offset_translation = centroid + rotated_offset

    arr = np.zeros(shape)
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
                return [0, 1]
            else:
                return [1, 0]
        else:
            return [1, 0]
    else:
        return [1, 0]


def sample_event_map(event_map,
                     centroid,
                     rotation,
                     translation,
                     shape,
                     ):
    return sample_map(event_map,
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


class EventMapDataset(Dataset):
    def __init__(self, table, shape=np.array([16, 16, 16])):
        self.table: pd.DataFrame = table
        self.sample_shape = shape

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        event_record = self.table.iloc[item]

        event = Event.from_record(event_record)

        event_centroid = np.array([event.x,
                                   event.y,
                                   event.z,
                                   ])

        event_map: gemmi.Grid = gemmi.read_ccp4_map(event.event_map_path)

        rotation = sample_rotation()
        translation = sample_translation()

        event_map_layer = sample_event_map(event_map,
                                           event_centroid,
                                           rotation,
                                           translation,
                                           self.sample_shape,
                                           )

        data = event_map_layer

        label = get_label(event)

        sample_dict = {"id": {"pandda_name": event.pandda_name,
                              "dtag": event.dtag,
                              "event_idx": event.event_idx,
                              },
                       "data": data,
                       "label": label,
                       }

        return sample_dict


def get_dataloader(table,
                   shape,
                   ):
    dataset: Dataset = EventMapDataset(table=table,
                                       shape=shape,
                                       )

    dataloader: DataLoader = DataLoader(dataset,
                                        batch_size=4,
                                        shuffle=True,
                                        num_workers=4,
                                        )

    return dataloader
