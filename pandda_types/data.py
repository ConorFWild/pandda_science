from typing import NamedTuple, Dict, Tuple, Union

from pathlib import Path

import pandas as pd

from biopandas.pdb import PandasPdb


class Event(NamedTuple):
    dtag: str
    event_idx: int
    occupancy: float
    analysed_resolution: float
    high_resolution: float
    interesting: bool
    ligand_placed: bool
    ligand_confidence: bool
    viewed: bool
    initial_model_path: Path
    data_path: Path
    final_model_path: Path
    event_map_path: Path
    actually_built: bool
    model_dir: Path
    pandda_dir: Path
    ligand_smiles_path: Path
    pandda_name: str
    x: float
    y: float
    z: float
    distance_to_ligand_model: float
    event_size: int

    @staticmethod
    def from_record(row):
        event: Event = Event(dtag=str(row["dtag"]),
                             event_idx=int(row["event_idx"]),
                             occupancy=float(row["occupancy"]),
                             analysed_resolution=float(row["analysed_resolution"]),
                             high_resolution=float(row["high_resolution"]),
                             interesting=row["interesting"],
                             ligand_placed=row["ligand_placed"],
                             ligand_confidence=row["ligand_confidence"],
                             viewed=row["viewed"],
                             initial_model_path=row["initial_model_path"],
                             data_path=row["data_path"],
                             model_dir=row["model_dir"],
                             pandda_dir=row["pandda_dir"],
                             pandda_name=row["pandda_name"],
                             final_model_path=row["final_model_path"],
                             event_map_path=row["event_map_path"],
                             actually_built=row["actually_built"],
                             ligand_smiles_path=row["ligand_smiles_path"],
                             x=row["x"],
                             y=row["y"],
                             z=row["z"],
                             distance_to_ligand_model=row["distance_to_ligand_model"],
                             event_size=row["event_size"]
                             )
        return event

    def to_record(self):
        record = {"pandda": self.pandda_dir,
                  "dtag": self.dtag,
                  "event_idx": self.event_idx,
                  "occupancy": self.occupancy,
                  "analysed_resolution": self.analysed_resolution,
                  "high_resolution": self.high_resolution,
                  "interesting": self.interesting,
                  "ligand_placed": self.ligand_placed,
                  "ligand_confidence": self.ligand_confidence,
                  "viewed": self.viewed,
                  "initial_model_path": self.initial_model_path,
                  "data_path": self.data_path,
                  "final_model_path": self.final_model_path,
                  "event_map_path": self.event_map_path,
                  "actually_built": self.actually_built,
                  "pandda_name": self.pandda_name,
                  "model_dir": self.model_dir,
                  "ligand_smiles_path": self.ligand_smiles_path,
                  "pandda_dir": self.pandda_dir,
                  "x": self.x,
                  "y": self.y,
                  "z": self.z,
                  "distance_to_ligand_model": self.distance_to_ligand_model,
                  "event_size": self.event_size,
                  }
        return record

    @staticmethod
    def from_pandda_path(pandda_path, model_dir):
        events = {}

        inspect_table_path = pandda_path / "analyses" / "pandda_analyse_events.csv"

        inspect_table = pd.read_csv(str(inspect_table_path))

        processed_datasets_path: Path = pandda_path / "processed_datasets"

        for idx, event_record in inspect_table.iterrows():
            dataset_path = processed_datasets_path / "{}".format(event_record["dtag"])
            modelled_structures_path = dataset_path / "{}".format("modelled_structures")
            initial_model_path = dataset_path / "{}-pandda-input.pdb".format(event_record["dtag"])
            data_path = dataset_path / "{}-pandda-input.mtz".format(event_record["dtag"])
            final_model_path = modelled_structures_path / "{}-pandda-model.pdb".format(event_record["dtag"])
            # final_model_path = modelled_structures_path / "fitted-v0001".format(event_record["dtag"])
            event_map_path = dataset_path / "{}-event_{}_1-BDC_{}_map.native.ccp4".format(event_record["dtag"],
                                                                                          event_record["event_idx"],
                                                                                          event_record["1-BDC"],
                                                                                          )
            ligand_smiles = get_ligand_smiles(dataset_path)

            actually_built = False

            distance_to_ligand_model = 0

            event: Event = Event(dtag=str(event_record["dtag"]),
                                 event_idx=int(event_record["event_idx"]),
                                 occupancy=event_record["1-BDC"],
                                 analysed_resolution=event_record["analysed_resolution"],
                                 high_resolution=0, # TODO: Not in PandDDA2 output
                                 interesting=event_record["Interesting"],
                                 ligand_placed=event_record["Ligand Placed"],
                                 ligand_confidence=event_record["Ligand Confidence"],
                                 viewed=event_record["Viewed"],
                                 initial_model_path=initial_model_path,
                                 data_path=data_path,
                                 model_dir=model_dir,
                                 pandda_dir=pandda_path,
                                 pandda_name=pandda_path.name,
                                 final_model_path=final_model_path,
                                 event_map_path=event_map_path,
                                 actually_built=actually_built,
                                 ligand_smiles_path=ligand_smiles,
                                 x=event_record["x"],
                                 y=event_record["y"],
                                 z=event_record["z"],
                                 distance_to_ligand_model=distance_to_ligand_model,
                                 event_size=event_record["cluster_size"],
                                 )

            events[(event.pandda_name, event.dtag, event.event_idx)] = event

        return events


def get_ligand_smiles(dataset_path):
    ligand_files_path = dataset_path / "ligand_files"
    smiles_paths = list(ligand_files_path.glob("*.smiles"))
    if len(smiles_paths) == 0:
        return "None"
    else:
        return smiles_paths[0]


def is_actually_built(final_model_path: Path):
    if final_model_path.is_file():
        model = PandasPdb().read_pdb(str(final_model_path))
        df = model.df["HETATM"]

        if len(df[df["residue_name"] == "LIG"]) != 0:
            return True
        else:
            return False
    else:
        return False


class PanDDA(NamedTuple):
    dir: Path
    events: Union[Dict[Tuple[str, int], Event], None]
    event_table_path: Path
    model_dir: Path
