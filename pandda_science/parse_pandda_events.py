from typing import Dict, Tuple, NamedTuple

import argparse
from pathlib import Path

import pandas as pd

from biopandas.pdb import PandasPdb

from pandda_types.data import PanDDA, Event


def get_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--root_dir",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-o", "--out_dir",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    args = parser.parse_args()

    return args


class Config(NamedTuple):
    root_dir: Path
    out_dir: Path


def get_config(args):
    config = Config(root_dir=Path(args.root_dir),
                    out_dir=Path(args.out_dir),
                    )

    return config


def get_panddas(root_dir: Path):
    # regex = "**/pandda_inspect_events.csv"
    regex = "*/*/processing/analysis/*/analyses/pandda_inspect_events.csv"

    paths = root_dir.glob(regex)

    panddas: Dict[Path, PanDDA] = {}

    for inspect_path in paths:
        pandda_dir: Path = inspect_path.parent.parent

        pandda = PanDDA(dir=pandda_dir,
                        events=None,
                        event_table_path=inspect_path,
                        )

        panddas[inspect_path] = pandda

    return panddas


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


def get_events(pandda: PanDDA):
    events: Dict[Tuple[str, int], Event] = {}

    inspect_table: pd.DataFrame = pd.read_csv(str(pandda.event_table_path))

    processed_datasets_path: Path = pandda.dir / "processed_datasets"

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

        actually_built = is_actually_built(final_model_path)

        event: Event = Event(dtag=str(event_record["dtag"]),
                             event_idx=int(event_record["event_idx"]),
                             occupancy=event_record["1-BDC"],
                             analysed_resolution=event_record["analysed_resolution"],
                             high_resolution=event_record["high_resolution"],
                             interesting=event_record["Interesting"],
                             ligand_placed=event_record["Ligand Placed"],
                             ligand_confidence=event_record["Ligand Confidence"],
                             viewed=event_record["Viewed"],
                             initial_model_path=initial_model_path,
                             data_path=data_path,
                             final_model_path=final_model_path,
                             event_map_path=event_map_path,
                             actually_built=actually_built,
                             x=event_record["x"],
                             y=event_record["y"],
                             z=event_record["z"],
                             )

        events[(event.dtag, event.event_idx)] = event

    return events


def get_panddas_table(panddas: Dict[Path, PanDDA]):
    records = []

    for path, pandda in panddas.items():

        for event_idx, event in pandda.events.items():
            record = {"pandda": pandda.dir,
                      "dtag": event.dtag,
                      "event_idx": event.event_idx,
                      "occupancy": event.occupancy,
                      "analysed_resolution": event.analysed_resolution,
                      "high_resolution": event.high_resolution,
                      "interesting": event.interesting,
                      "ligand_placed": event.ligand_placed,
                      "ligand_confidence": event.ligand_confidence,
                      "viewed": event.viewed,
                      "initial_model_path": event.initial_model_path,
                      "data_path": event.data_path,
                      "final_model_path": event.final_model_path,
                      "event_map_path": event.event_map_path,
                      "actually_built": event.actually_built,
                      "x": event.x,
                      "y": event.y,
                      "z": event.z,
                      }

            records.append(record)

    df: pd.DataFrame = pd.DataFrame(records)

    return df


def output_table(df: pd.DataFrame, output_path: Path):
    df.to_csv(str(output_path))


if __name__ == "__main__":

    args = get_args()

    config: Config = get_config(args)

    print("Looking for pandda inspect tables in: {}".format(config.root_dir))
    panddas: Dict[Path, PanDDA] = get_panddas(config.root_dir)
    print("\tFound: {} panddas".format(len(panddas)))

    print("Looking for event from found event tables...")
    panddas_with_events: Dict[Path, PanDDA] = {}
    for pandda_path, pandda in panddas.items():
        print("\tGetting events from: {}".format(pandda_path))
        events: Dict[Tuple[str, int], Event] = get_events(pandda)
        print("\t\tFound: {} events".format(len(events)))
        print(
            "\t\tActually built: {}".format(len([event for event_id, event in events.items() if event.actually_built])))

        panddas_with_events[pandda_path] = PanDDA(dir=pandda.dir,
                                                  events=events,
                                                  event_table_path=pandda.event_table_path,
                                                  )

    print("Constructing pandda table...")
    pandda_table: pd.DataFrame = get_panddas_table(panddas_with_events)

    print("Outputting pandda table to: {}".format(config.out_dir / "event_table.csv"))
    output_table(pandda_table,
                 config.out_dir / "event_table.csv",
                 )
