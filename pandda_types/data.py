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
    x: float
    y: float
    z: float


class PanDDA(NamedTuple):
    dir: Path
    events: Union[Dict[Tuple[str, int], Event], None]
    event_table_path: Path
