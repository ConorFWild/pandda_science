import os
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import seaborn
from matplotlib import pyplot as plt

import gemmi

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
                                  RhofitResultsFile,
                                  )


class Config:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-o", "--old_pandda_dir",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        parser.add_argument("-n", "--new_pandda_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        args = parser.parse_args()

        self.old_pandda_dir = Path(args.old_pandda_dir)
        self.new_pandda_dir = Path(args.new_pandda_dir)


class TableEvent:
    def __init__(self,
                 dtag,
                 event_idx,
                 event_dir,
                 pdb_path,
                 mtz_path,
                 smiles_path,
                 occupancy,
                 event_map_path,
                 ):
        self.dtag = dtag
        self.event_idx = event_idx
        self.event_dir = event_dir
        self.pdb_path = pdb_path
        self.mtz_path = mtz_path
        self.smiles_path = smiles_path
        self.occupancy = occupancy
        self.event_map_path = event_map_path

    @staticmethod
    def from_row(row, pandda_dir):
        dtag = row["dtag"]
        event_idx = row["event_idx"]

        processed_datasets_dir = pandda_dir / PANDDA_PROCESSED_DATASETS_DIR

        event_dir = processed_datasets_dir / dtag

        pdb_path = event_dir / PANDDA_PDB_FILE.format(dtag)
        mtz_path = event_dir / PANDDA_MTZ_FILE.format(dtag)
        smiles_path = get_ligand_smiles(event_dir)


        occupancy = row["1-BDC"]
        event_map_path = event_dir / PANDDA_EVENT_MAP_FILE.format(dtag, event_idx, occupancy)

        return TableEvent(dtag,
                          event_idx,
                          event_dir,
                          pdb_path,
                          mtz_path,
                          smiles_path,
                          occupancy,
                          event_map_path,
                          )


def get_ligand_smiles(pandda_event_dir):
    compound_dir = pandda_event_dir / "ligand_files"

    smiles_paths = compound_dir.glob("*.smiles")
    smiles_paths_list = list(smiles_paths)

    if len(smiles_paths_list) > 0:
        return Path(min([str(ligand_path) for ligand_path in smiles_paths_list if ligand_path.name != "tmp.smiles"],
                        key=len)
                    )
    else:
        raise Exception("No smiles found! Smiles list is: {}".format(smiles_paths_list))

    # ligand_pdbs = list(compound_dir.glob("*.pdb"))
    # ligand_pdb_strings = [str(ligand_path) for ligand_path in ligand_pdbs if ligand_path.name != "tmp.pdb"]
    # if len(ligand_pdb_strings) > 0:
    #     shortest_ligand_path = min(ligand_pdb_strings,
    #                                key=len,
    #                                )
    #     return Path(shortest_ligand_path)




# def get_events(path):
#     events = {}
#     event_table = pd.read_csv(str(path))
#     for idx, event_row in event_table.iterrows():
#         pandda_processed_dir = Path(event_row["event_map_path"]).parent
#         print(pandda_processed_dir)
#         if pandda_processed_dir.exists():
#             try:
#                 event_row["ligand_smiles_path"] = get_ligand_smiles(Path(event_row["event_map_path"]).parent)
#                 print("\tLigand smiles path: {}".format(event_row["ligand_smiles_path"]))
#                 event = Event.from_record(event_row)
#                 events[(event.pandda_name, event.dtag, event.event_idx)] = event
#
#             except Exception as e:
#                 print(e)
#                 continue
#
#         else:
#             continue
#
#     return events


def get_event_table(old_pandda_dir):
    event_table_path = old_pandda_dir / "analyses" / "pandda_inspect_events.csv"

    return pd.read_csv(str(event_table_path))


def get_events(event_table,
               old_pandda_dir,
               ):
    events = {}
    for index, row in event_table.iterrows():
        pandda_name = old_pandda_dir.name

        try:
            event = TableEvent.from_row(row,
                                        old_pandda_dir,
                                        )
            events[(event.dtag, event.event_idx)] = event
        except Exception as e:
            print("\t{}".format(e))

    return events


def try_make_dir(path):
    if not path.exists():
        os.mkdir(str(path))


def try_copy(old_path, new_path):
    if not new_path.exists():
        shutil.copyfile(str(old_path),
                        str(new_path),
                        )


def try_copy_autobuild_files(processed_dataset_dir,
                             event,
                             ):
    # Modelled folder
    modelled_datasets_dir = processed_dataset_dir / PANDDA_MODELLED_STRUCTURES_DIR
    print("\t\tMaking modelled datasets dir at {}".format(modelled_datasets_dir))
    try_make_dir(modelled_datasets_dir)

    # ligand files
    ligands_dir = processed_dataset_dir / PANDDA_LIGAND_FILES_DIR
    print("\t\tMaking ligands dir at {}".format(ligands_dir))
    try_make_dir(ligands_dir)

    # Copy pdb
    new_pdb_path = processed_dataset_dir / event.pdb_path.name
    print("\t\tTrying to copy pdb from {} to {}".format(event.pdb_path,
                                                        new_pdb_path,
                                                        )
          )
    try_copy(event.pdb_path,
             new_pdb_path,
             )

    # Copy mtz
    new_mtz_path = processed_dataset_dir / event.mtz_path.name
    print("\t\tTrying to copy pdb from {} to {}".format(event.mtz_path,
                                                        new_mtz_path,
                                                        )
          )
    try_copy(event.mtz_path,
             new_mtz_path,
             )

    # Copy ligand
    new_smiles_path = processed_dataset_dir / PANDDA_LIGAND_FILES_DIR / event.smiles_path.name
    print("\t\tTrying to copy pdb from {} to {}".format(event.smiles_path,
                                                        new_smiles_path,
                                                        )
          )
    try_copy(event.smiles_path,
             new_smiles_path,
             )

    # Copy event map
    new_event_map_path = processed_dataset_dir / PANDDA_EVENT_MAP_FILE.format(event.dtag,
                                                                              event.event_idx,
                                                                              event.occupancy,
                                                                              )
    print("\t\tTrying to copy event map from {} to {}".format(event.event_map_path,
                                                        new_event_map_path,
                                                        )
          )
    try_copy(event.event_map_path,
             new_event_map_path,
             )


def copy_pandda(pandda_events,
                old_pandda_dir,
                new_pannda_dir,
                ):
    print("\tTrying to make new pandda dir at: {}".format(new_pannda_dir))
    try_make_dir(new_pannda_dir)

    processed_datasets_path = new_pannda_dir / "processed_datasets"
    print("\tTrying to make new processed datasets dir at: {}".format(processed_datasets_path))
    try_make_dir(processed_datasets_path)

    analyses_dir = new_pannda_dir / PANDDA_ANALYSES_DIR
    print("\tMaking analyses dir")
    try_make_dir(analyses_dir)

    old_pandda_inspect_events_path = old_pandda_dir / PANDDA_ANALYSES_DIR / PANDDA_INSPECT_EVENTS_PATH
    new_pandda_inspect_events_path = new_pannda_dir / PANDDA_ANALYSES_DIR / PANDDA_INSPECT_EVENTS_PATH
    try_copy(old_pandda_inspect_events_path,
             new_pandda_inspect_events_path,
             )

    for event_id, event in pandda_events.items():
        print("\tTrying to copy data for: {}".format(event_id))
        processed_dataset_path = processed_datasets_path / event.dtag
        try_make_dir(processed_dataset_path)
        try_copy_autobuild_files(processed_dataset_path,
                                 event,
                                 )


def autobuild_pandda(new_pannda_dir):
    module = "module load gcc/4.9.3"
    env = "source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_clipper_no_mkl"
    python = "python"
    autobuild = "/dls/science/groups/i04-1/conor_dev/pandda_science/pandda_autobuilding/autobuild_pandda.py"
    input = new_pannda_dir
    overwrite = "1"

    command = "{}; {}; {} {} -i {} -o {} -p 1"
    formated_command = command.format(module,
                                      env,
                                      python,
                                      autobuild,
                                      input,
                                      overwrite,
                                      )

    print("\tPanDDA submit command is: {}".format(formated_command))
    submit_proc = subprocess.Popen(str(formated_command),
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   )
    stdout, stderr = submit_proc.communicate()

    return stdout, stderr


def parse_results(new_pandda_dir,
                  events,
                  ):
    results = {}
    for event_id, event in events.items():

        results_path = new_pandda_dir / PANDDA_PROCESSED_DATASETS_DIR / event.dtag / RHOFIT_DIR / RHOFIT_RESULTS_FILE

        if results_path.exists():
            rhofit_results_file = RhofitResultsFile.from_file(results_path)
            results[(event.pandda_name, event.dtag, event.event_idx)] = rhofit_results_file
        else:
            print("\tNo result at {}".format(results_path))

    return results


def get_cumulative_hits_event_size(pandda_events):
    sorted_events = sorted(pandda_events.values(),
                           key=lambda event: event.event_size,
                           )

    num_hits = 0
    hits = []
    for event in sorted_events:
        if event.modelled is True:
            num_hits = num_hits + 1
            hits.append(num_hits)

    return hits


def get_cumulative_hits_rscc(pandda_events,
                             results,
                             ):
    sorted_results = sorted(results.values(),
                            key=lambda result: result.rscc,
                            )

    num_hits = 0
    hits = []
    for result in results:
        dtag = result.dtag
        event_idx = result.event_idx
        event_id = (dtag, event_idx)

        if pandda_events[event_id].modelled is True:
            num_hits = num_hits + 1
            hits.append(num_hits)

    return hits


def get_enritchment_table(cumulative_hits_list):
    enritchments = []
    for i, num_hits in enumerate(cumulative_hits_list):
        index = i + 1
        enritchment = num_hits / index

        record = {"index": index,
                  "enritchment": enritchment,
                  }

        enritchments.append(record)

    return pd.DataFrame(enritchments)


def main():
    config = Config()

    event_table = get_event_table(config.old_pandda_dir)
    print(event_table.head(10))

    if len(event_table) == 0:
        raise Exception("No events pulled!")

    print("Getting events...")
    pandda_events = get_events(event_table,
                               config.old_pandda_dir)

    print("Copying PanDDA...")
    copy_pandda(pandda_events,
                config.old_pandda_dir,
                config.new_pandda_dir,
                )

    print("Autobuilding pandda...")
    stdout, stderr = autobuild_pandda(config.new_pandda_dir)
    print(stdout)
    print(stderr)

    print("Parsing results...")
    results = parse_results(config.new_pandda_dir,
                            pandda_events,
                            )

    print("Getting cumulative hits")
    cumulative_hits_event_size = get_cumulative_hits_event_size(pandda_events)

    cumulative_hits_rscc = get_cumulative_hits_rscc(pandda_events,
                                                    results,
                                                    )

    print("Getting enritchment tables...")
    enritchment_table_event_size = get_enritchment_table(cumulative_hits_event_size)
    print(enritchment_table_event_size)

    enritchment_table_rscc = get_enritchment_table(cumulative_hits_rscc)
    print(enritchment_table_rscc)

    print("Plotting cumulative hits")
    plot_cumulative_hits(cumulative_hits_event_size,
                         cumulative_hits_rscc,
                         config.new_pandda_dir / CUMULATIVE_HITS_PLOT_FILE,
                         )


if __name__ == "__main__":
    main()
