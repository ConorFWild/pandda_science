import os
import argparse
from pathlib import Path

import pandas as pd

import luigi

from pandda_autobuilding.constants import *
from pandda_types.data import Event
from pandda_types.process import (Rhofit,
                                  AutobuildingResultRhofit,
                                  Elbow,
                                  MapToMTZ,
                                  Strip,
                                  Graft,
                                  QSub,
                                  )


def get_ligand_smiles(pandda_event_dir):
    compound_dir = pandda_event_dir / "ligand_files"

    ligand_pdbs = list(compound_dir.glob("*.pdb"))
    ligand_pdb_strings = [str(ligand_path) for ligand_path in ligand_pdbs if ligand_path.name != "tmp.pdb"]
    if len(ligand_pdb_strings) > 0:
        shortest_ligand_path = min(ligand_pdb_strings,
                                   key=len,
                                   )
        return Path(shortest_ligand_path)

    smiles_paths = compound_dir.glob("*.smiles")
    smiles_paths_list = list(smiles_paths)

    if len(smiles_paths_list) > 0:
        return Path(min([str(ligand_path) for ligand_path in smiles_paths_list if ligand_path.name != "tmp.smiles"],
                        key=len)
                    )
    else:
        raise Exception("No smiles found! Smiles list is: {}".format(smiles_paths_list))


def get_events(path):
    events = []
    event_table = pd.read_csv(str(path))
    for idx, event_row in event_table.iterrows():
        if event_row["actually_built"] is True:
            pandda_processed_dir = Path(event_row["event_map_path"]).parent
            print(pandda_processed_dir)
            if pandda_processed_dir.exists():
                try:
                    event_row["ligand_smiles_path"] = get_ligand_smiles(Path(event_row["event_map_path"]).parent)
                    print("\tLigand smiles path:{}".format(event_row["ligand_smiles_path"]))
                    event = Event.from_record(event_row)
                    events.append(event)

                except Exception as e:
                    print(e)
                    continue

        else:
            continue

    return events


class Config:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-i", "--event_table_path",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        parser.add_argument("-o", "--out_dir_path",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        args = parser.parse_args()

        self.event_table_path = args.event_table_path
        self.out_dir_path = Path(args.out_dir_path)


def load_available_results_events(events, out_dir_path):
    results = {}
    for event in events:
        results_normal_path = Path(out_dir_path) / BUILD_DIR_PATTERN.format(
                                                       pandda_name=event.pandda_name,
                                                       dtag=event.dtag,
                                                       event_idx=event.event_idx,
                                                       ) / RHOFIT_RESULT_JSON_FILE
        if results_normal_path.exists():
            result = AutobuildingResultRhofit.from_json(results_normal_path)
            results[(event.pandda_name, event.dtag, event.event_idx)] = result
        else:
            print("\tNo result at {}".format(results_normal_path))

            
def load_available_results_events(events, out_dir_path):
    results = {}
    for event in events:
        results_normal_path = Path(out_dir_path) / BUILD_DIR_PATTERN.format(
                                                       pandda_name=event.pandda_name,
                                                       dtag=event.dtag,
                                                       event_idx=event.event_idx,
                                                       ) / RHOFIT_NORMAL_RESULT_JSON_FILE
        if results_normal_path.exists():
            result = AutobuildingResultRhofit.from_json(results_normal_path)
            results[(event.pandda_name, event.dtag, event.event_idx)] = result
        else:
            print("\tNo result at {}".format(results_normal_path))


if __name__ == "__main__":
    print("Geting Config...")
    config = Config()

    print("Getting event table...")
    events = get_events(config.event_table_path)
    print("\tGot {} events!".format(len(events)))

    results_event = load_available_results_events(events)
    print("\t {} event results available".format(len(results_event)))
    results_normal = load_available_results_normal(events)
    print("\t {} normal results available".format(results_normal))



