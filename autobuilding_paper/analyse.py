import argparse
import json
from pprint import PrettyPrinter

from pathlib import Path

import pandas as pd

from pandda_types import logs
from pandda_types.process import QSub

from autobuilding_paper.lib import ReferenceStructures
from autobuilding_paper.pandda_events import PanDDAEventDistances
from autobuilding_paper.autobuild_rmsd_table import AutobuildRMSDTable
from autobuilding_paper.results import SystemTable, PanDDAResults, AutobuildResults
from autobuilding_paper.ranking import PanDDARanking, Enritchment
from autobuilding_paper.constants import *


class Config:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-i", "--system_file",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        parser.add_argument("-p", "--panddas_file",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        parser.add_argument("-a", "--autobuild_file",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        parser.add_argument("-o", "--output_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        args = parser.parse_args()

        self.system_file = Path(args.system_file)
        self.panddas_file = Path(args.panddas_file)
        self.autobuild_file = Path(args.autobuild_file)
        self.output_dir = Path(args.output_dir)


def to_json(dictionary, path):
    with open(str(path), "w") as f:
        json.dump(f,
                  dictionary,
                  )


def main():
    printer = PrettyPrinter(depth=1)
    config = Config()

    system_table = SystemTable.from_json(config.system_file)
    printer.pprint(system_table)

    pandda_table = PanDDAResults.from_json(config.panddas_file)
    printer.pprint(pandda_table)

    autobuilding_table = AutobuildResults.from_json(config.autobuild_file)
    printer.pprint(autobuilding_table)

    results = {}
    for pandda_id, pandda_info in pandda_table.to_dict().items():
        logs.LOG[pandda_id] = {}
        pandda_dir = Path(pandda_info["out_dir"])

        # Closest event: how good is PanDDA2
        dataset_events = PanDDAEventDistances.from_dir(pandda_dir)
        logs.LOG[pandda_id]["event_distances"] = dataset_events

        # Ligand RMSD: how good is autobuilding
        ligand_rmsds = AutobuildRMSDTable.from_directory(pandda_dir)
        logs.LOG[pandda_id]["ligand_rmsds"] = ligand_rmsds

        # Ranking
        references = ReferenceStructures.from_dir(pandda_dir)
        naive_ranking = PanDDARanking.from_pandda_dir(pandda_dir)
        autobuilding_ranking = PanDDARanking.from_autobuild_rscc(autobuilding_table[pandda_id])
        naive_enritchment = Enritchment.from_ranking(naive_ranking,
                                                     references,
                                                     )
        autobuilding_enritchment = Enritchment.from_ranking(autobuilding_ranking,
                                                        references,
                                                        )
        logs.LOG[pandda_id]["naive"] = naive_enritchment.enritchment
        logs.LOG[pandda_id]["autobuilding"] = autobuilding_enritchment.enritchment

        printer.pprint(logs.LOG.dict)

    to_json(results,
            config.autobuild_file,
            )


if __name__ == "__main__":
    main()
