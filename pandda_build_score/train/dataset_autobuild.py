from typing import NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd

import gemmi

from torch.utils.data import Dataset, DataLoader


class Build(NamedTuple):
    system: str
    dtag: str
    event_idx: int
    resolution: float
    ligand_build_path: Path
    stripped_receptor_path: Path
    x: float
    y: float
    z: float
    data_path: Path
    human_build: bool
    rmsd: float


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


def get_human_build_models(path: Path,
                           event_centroid,
                           ):
    struc = gemmi.read_structure(path)

    lig_residues = []

    for model in struc:
        for chain in model:
            for residue in chain:
                if residue.name == 'LIG':
                    lig_residues.append(residue)

    distances = []
    for residue in lig_residues:
        distances.append(get_distance(residue,
                                      event_centroid,
                                      ))
    closest_lig_to_event = distances[np.argmin(distances)]

    return closest_lig_to_event


def get_receptor_model(path):
    return gemmi.read_structure(path)


def get_ligand_model(path):
    return gemmi.read_structure(path)


def sample_rotation():
    return np.random.rand(3) * np.pi * 2


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


def get_label(human_build,
              rmsd,
              event_centroid,
              ligand_model,
              ):
    if human_build:
        return [1, 0, 0, 0, 0]
    elif rmsd < 0.5:
        return [1, 0, 0, 0, 0]
    elif rmsd < 1:
        return [0, 1, 0, 0, 0]
    elif rmsd < 2:
        return [0, 0, 1, 0, 0]
    elif rmsd < 5:
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]


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


def get_element_key():
    return {"C": 0,
            "O": 1,
            "N": 2,
            "S": 3,
            }


def clone_grid(grid):
    new_grid = gemmi.FloatGrid(grid.nu, grid.nv, grid.nw)
    new_grid.spacegroup = grid.spacegroup
    new_grid.set_unit_cell(grid.unit_cell)
    return new_grid


def sample_receptor_layers(event_map,
                           receptor_model,
                           event_centroid,
                           rotation,
                           translation,
                           shape,
                           ):
    element_key = get_element_key()

    layers = [clone_grid(event_map)
              for element
              in element_key
              ]
    marks = gemmi.subcells.find_atoms(gemmi.Position(event_centroid[0],
                                                     event_centroid[1],
                                                     event_centroid[2],
                                                     ),
                                      '\0',
                                      radius=10,
                                      )
    for atom_mark in marks:
        cra = atom_mark.to_cra(receptor_model)
        pos = cra.atom.pos
        elm = cra.atom.element

        layers[element_key[elm.name]].set_points_around(pos,
                                                        radius=1,
                                                        value=1,
                                                        )

    samples = [sample_map(layer,
                          event_centroid,
                          rotation,
                          translation,
                          shape,
                          )
               for layer
               in layers]

    return samples


def sample_ligand_layers(event_map,
                         ligand_residue,
                         event_centroid,
                         rotation,
                         translation,
                         shape,
                         ):
    element_key = get_element_key()

    layers = [clone_grid(event_map)
              for element
              in element_key
              ]
    for atom in ligand_residue:
        pos = atom.pos
        elm = atom.element

        layers[element_key[elm.name]].set_points_around(pos,
                                                        radius=1,
                                                        value=1,
                                                        )

    samples = [sample_map(layer,
                          event_centroid,
                          rotation,
                          translation,
                          shape,
                          )
               for layer
               in layers]

    return samples


class BuildScoreDataset(Dataset):
    def __init__(self, csv_path, shape=np.array([16, 16, 16])):
        self.csv_path: Path = csv_path
        self.table: pd.DataFrame = pd.read_csv(self.csv_path)

        self.sample_shape = shape

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        build_record = self.table.iloc[item]

        build: Build = Build(system=build_record["system"],
                             dtag=build_record["dtag"],
                             event_idx=build_record["event_idx"],
                             resolution=float(build_record["resolution"]),
                             ligand_build_path=Path(build_record["ligand_build_path"]),
                             stripped_receptor_path=Path(build_record["stripped_receptor_path"]),
                             x=float(build_record["x"]),
                             y=float(build_record["y"]),
                             z=float(build_record["z"]),
                             data_path=Path(build_record["data_path"]),
                             human_build=bool(build_record["human_build"]),
                             rmsd=float(build_record["rmsd"]),
                             )

        event_centroid = np.array([build.x,
                                   build.y,
                                   build.z,
                                   ])

        event_map: gemmi.Grid = gemmi.read_ccp4_map(build.data_path)

        if build.human_build:
            receptor_model, ligand_model = get_human_build_models(build.ligand_build_path,
                                                                  event_centroid=event_centroid,
                                                                  )
        else:
            receptor_model = get_receptor_model(build.stripped_receptor_path)
            ligand_model = get_ligand_model(build.ligand_build_path)

        rotation = sample_rotation()
        translation = sample_translation()

        event_map_layer = sample_event_map(event_map,
                                           event_centroid,
                                           rotation,
                                           translation,
                                           self.shape,
                                           )
        receptor_layers = sample_receptor_layers(event_map,
                                                 receptor_model,
                                                 event_centroid,
                                                 rotation,
                                                 translation,
                                                 self.shape,
                                                 )
        ligand_layers = sample_ligand_layers(event_map,
                                             ligand_model,
                                             event_centroid,
                                             rotation,
                                             translation,
                                             self.shape,
                                             )

        data = np.concatenate([event_map_layer] + receptor_layers + ligand_layers)

        label = get_label(build.human_build,
                          build.rmsd,
                          event_centroid,
                          ligand_model,
                          )

        return {"id": {"dtag": build.dtag, "event_idx": build.event_idx}, "data": data, "label": label}


def get_dataloader(csv_path: Path,
                   shape,
                   ):
    dataset: Dataset = BuildScoreDataset(csv_path,
                                         shape=shape,
                                         )

    dataloader: DataLoader = DataLoader(dataset,
                                        batch_size=4,
                                        shuffle=True,
                                        num_workers=4,
                                        )

    return dataloader
