from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset, DataLoader

from pandda_3.types.data_types import Event


class PanDDAEventsDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path: Path = csv_path
        self.table: pd.DataFrame = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        event_record = self.table.iloc[item]

        event: Event = Event(dtag=str(event_record["dtag"]),
                             event_idx=int(event_record["event_idx"]),
                             occupancy=event_record["occupancy"],
                             analysed_resolution=event_record["analysed_resolution"],
                             high_resolution=event_record["high_resolution"],
                             interesting=event_record["interesting"],
                             ligand_placed=event_record["ligand_placed"],
                             ligand_confidence=event_record["ligand_confidence"],
                             viewed=event_record["Viewed"],
                             initial_model_path=event_record["initial_model_path"],
                             data_path=event_record["data_path"],
                             final_model_path=event_record["final_model_path"],
                             x=event_record["x"],
                             y=event_record["y"],
                             z=event_record["z"],
                             )

        return event


def get_dataloader(csv_path: Path):
    dataset: Dataset = PanDDAEventsDataset(csv_path)

    dataloader: DataLoader = DataLoader(dataset,
                                        batch_size=4,
                                        shuffle=True,
                                        num_workers=4,
                                        )

    return dataloader
