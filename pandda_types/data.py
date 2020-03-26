from typing import NamedTuple, Dict, Tuple, Union

from pathlib import Path


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
                  }
        return record


class PanDDA(NamedTuple):
    dir: Path
    events: Union[Dict[Tuple[str, int], Event], None]
    event_table_path: Path
    model_dir: Path

