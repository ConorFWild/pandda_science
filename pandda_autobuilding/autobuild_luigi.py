import os
import argparse
from pathlib import Path

import pandas as pd

import luigi

from pandda_types.data import Event
from pandda_types.process import (Rhofit,
                                  AutobuildingResultRhofit,
                                  Elbow,
                                  MapToMTZ,
                                  Strip,
                                  Graft,
                                  QSub,
                                  )

STRIPPED_RECEPTOR_FILE = "stripped_receptor.pdb"
LIGAND_FILE = "autobuilding_ligand.cif"
EVENT_MTZ_FILE = "event.mtz"
GRAFTED_MTZ_FILE = "grafted.mtz"
RHOFIT_DIR = "rhofit"
RHOFIT_RESULTS_FILE = "results.txt"
RHOFIT_RESULT_JSON_FILE = "result.json"
RHOFIT_BEST_MODEL_FILE = "best.pdb"
RSCC_TABLE_FILE = "rscc_table.csv"

BUILD_DIR_PATTERN = "{pandda_name}_{dtag}_{event_idx}"


class ElbowTask(luigi.Task):
    event = luigi.Parameter()
    out_dir_path = luigi.Parameter()

    def requires(self):
        return DirSetupTask(
            out_dir_path=self.out_dir_path / BUILD_DIR_PATTERN.format(pandda_name=self.event.pandda_name,
                                                                      dtag=self.event.dtag,
                                                                      event_idx=self.event.event_idx,
                                                                      )
            )

    def run(self):
        command = Elbow(self.out_dir_path,
                        self.event.ligand_smiles_path,
                        )

        QSub(command,
             self.out_dir_path / "elbow_command.sh",
             )()

    def output(self):
        return luigi.LocalTarget(str(self.out_dir_path / LIGAND_FILE))


class CCP4ToMTZTask(luigi.Task):
    event = luigi.Parameter()
    out_dir_path = luigi.Parameter()

    def requires(self):
        return DirSetupTask(
            out_dir_path=self.out_dir_path / BUILD_DIR_PATTERN.format(pandda_name=self.event.pandda_name,
                                                                      dtag=self.event.dtag,
                                                                      event_idx=self.event.event_idx,
                                                                      )
            )

    def run(self):
        command = MapToMTZ(self.event.event_map_path,
                           self.out_dir_path / EVENT_MTZ_FILE,
                           self.event.analysed_resolution,
                           )

        QSub(command,
             self.out_dir_path / "ccp4_to_mtz_command.sh",
             )()

    def output(self):
        return luigi.LocalTarget(str(self.out_dir_path / EVENT_MTZ_FILE))


class GraftTask(luigi.Task):
    out_dir_path = luigi.Parameter()
    event = luigi.Parameter()

    def requires(self):
        return CCP4ToMTZTask(event=self.event,
                             out_dir_path=self.out_dir_path,
                             )

    def run(self):
        Graft(self.event.data_path,
              self.out_dir_path / EVENT_MTZ_FILE,
              self.out_dir_path / GRAFTED_MTZ_FILE,
              )

    def output(self):
        return luigi.LocalTarget(str(self.out_dir_path / GRAFTED_MTZ_FILE))


class DirSetupTask(luigi.Task):
    out_dir_path = luigi.Parameter()

    def run(self):
        if not self.out_dir_path.exists():
            os.mkdir(str(self.out_dir_path))

    def output(self):
        return luigi.LocalTarget(self.out_dir_path)


class StripTask(luigi.Task):
    event = luigi.Parameter()
    out_dir_path = luigi.Parameter()

    def requires(self):
        return DirSetupTask(
            out_dir_path=self.out_dir_path / BUILD_DIR_PATTERN.format(pandda_name=self.event.pandda_name,
                                                                      dtag=self.event.dtag,
                                                                      event_idx=self.event.event_idx,
                                                                      )
            )

    def run(self):
        Strip(self.event.initial_model_path,
              [self.event.x, self.event.y, self.event.z],
              self.out_dir_path / STRIPPED_RECEPTOR_FILE,
              )

    def output(self):
        return luigi.LocalTarget(str(self.out_dir_path / STRIPPED_RECEPTOR_FILE))


class AutobuildRhofitTask(luigi.Task):
    event = luigi.Parameter()
    out_dir_path = luigi.Parameter()

    def requires(self):
        return [ElbowTask(event=self.event,
                          out_dir_path=self.out_dir_path,
                          ),
                GraftTask(event=self.event,
                          out_dir_path=self.out_dir_path,
                          ),
                StripTask(event=self.event,
                          out_dir_path=self.out_dir_path,
                          ),
                ]

    def run(self):
        command = Rhofit(out_dir_path=self.event.out_dir_path,
                         mtz_path=self.event.event_mtz_path,
                         ligand_path=self.event.ligand_path,
                         receptor_path=self.event.intial_receptor_path,
                         )
        QSub(command,
             self.out_dir_path,
             )

    def output(self):
        return luigi.LocalTarget(str(self.out_dir_path / RHOFIT_DIR / RHOFIT_BEST_MODEL_FILE))


class ParseResultsRhofit(luigi.Task):
    event = luigi.Parameter()
    out_dir_path = luigi.Parameter()

    def requires(self):
        return AutobuildRhofitTask(event=self.event,
                                   out_dir_path=self.out_dir_path,
                                   )

    def run(self):
        rhofit_dir = self.out_dir_path / RHOFIT_DIR
        result = AutobuildingResultRhofit.from_output(rhofit_dir,
                                                      self.event.pandda_name,
                                                      self.event.dtag,
                                                      self.event.event_idx,
                                                      )
        result.to_json(self.out_dir_path / RHOFIT_RESULT_JSON_FILE)

    def output(self):
        return luigi.LocalTarget(self.out_dir_path / RHOFIT_RESULT_JSON_FILE)


class ResultsTable(luigi.Task):
    out_dir_path = luigi.Parameter()
    events = luigi.Parameter()

    def requires(self):
        return [ParseResultsRhofit(event=event,
                                   out_dir_path=self.out_dir_path,
                                   )
                for event
                in events
                ]

    def run(self):
        results_paths = self.out_dir_path.glob("**/result_*.json")
        results = [AutobuildingResultRhofit.from_json(results_path)
                   for results_path
                   in results_paths
                   ]

        records = []
        for result in results:
            record = {}
            record["pandda_name"] = result.pandda_name
            record["dtag"] = result.dtag
            record["event_idx"] = result.event_idx
            record["rscc"] = result.rscc
            records.append(record)

        table = pd.DataFrame(records)
        table.to_csv(self.out_dir_path / RSCC_TABLE_FILE)

    def output(self):
        return luigi.LocalTarget(self.out_dir_path / RSCC_TABLE_FILE)


########################################################################################################################

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
            pandda_processed_dir = Path(event_row["event_map_path"])
            print(pandda_processed_dir)
            if pandda_processed_dir.exists():
                event_row["ligand_smiles_path"] = get_ligand_smiles(Path(event_row["event_map_path"]).parent)
                event = Event.from_record(event_row)
                events.append(event)
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
        self.out_dir_path = args.out_dir_path


if __name__ == "__main__":
    print("Geting Config...")
    config = Config()

    print("Getting event table...")
    events = get_events(config.event_table_path)
    print("\tGot {} events!".format(len(events)))

    tasks = [ResultsTable(out_dir_path=config.out_dir_path,
                          events=events,
                          )
             ]

    luigi.build(tasks,
                workers=100,
                local_scheduler=True,
                )
