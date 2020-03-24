from typing import NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from pandda_3.types.data_types import Event

from pandda_types import structure


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


def get_human_build_models(path: Path):
    struc = structure.Structure(path)

    lig_residues = []

    for model in struc.structure:
        for chain in model:
            for residue in chain:
                if residue.name == 'LIG':
                    lig_residues.append(residue)


class BuildScoreDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path: Path = csv_path
        self.table: pd.DataFrame = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        build_record = self.table.iloc[item]

        build: Build = Build(...)

        event_map: gemmi.Grid = get_event_map()
        if build.human_build:
            receptor_model, ligand_model = get_human_build_models()
        else:
            receptor_model = get_receptor_model()
            ligand_model = get_ligand_model()

        rotation = sample_rotation
        translation = sample_translation

        event_map_layer = sample_event_map(build)
        receptor_layers = sample_receptor_layers(receptor_model,
                                                 )
        ligand_layers = sample_ligand_layers()

        data = np.concatenate([event_map_layer] + receptor_layers + ligand_layers)

        label = get_label(build.human_build,
                          build.rmsd,
                          [build.x, build.y, build.z],
                          ligand_model,
                          )

        return {"data": data, "label": label}


def get_dataloader(csv_path: Path):
    dataset: Dataset = BuildScoreDataset(csv_path)

    dataloader: DataLoader = DataLoader(dataset,
                                        batch_size=4,
                                        shuffle=True,
                                        num_workers=4,
                                        )

    return dataloader
