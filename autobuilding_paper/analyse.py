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
    printer.pprint("Got system table")

    pandda_table = PanDDAResults.from_json(config.panddas_file)
    printer.pprint("Got pandda table")

    autobuilding_table = AutobuildResults.from_json(config.autobuild_file)
    rsccs = {dtag: max([autobuilding_table.results[dtag][event_id]["rscc"]
                        for event_id
                        in autobuilding_table.results[dtag]])
             for dtag
             in autobuilding_table.results
             }
    printer.pprint("Got autobuilding table")
    # printer.pprint(autobuilding_table)

    results = {}
    for pandda_id, pandda_info in pandda_table.to_dict().items():
        # try:
        printer.pprint("##### Analysing {} #####".format(pandda_id))
        # try:
        logs.LOG[pandda_id] = {}
        pandda_dir = Path(pandda_info.out_dir)

        # Closest event: how good is PanDDA2
        printer.pprint("# Analysing event distances")
        dataset_events = PanDDAEventDistances.from_dir(pandda_dir)
        printer.pprint(dataset_events.distances)
        # logs.LOG[pandda_id]["event_distances"] = dataset_events

        # RSCCs
        printer.pprint("# Analysing ligand rsccs")
        rscc_large_mask = {dtag: rscc for dtag, rscc in rsccs.items() if rscc > 0.7}
        printer.pprint(f"rscc_large_mask: {len(rscc_large_mask)}")
        printer.pprint(autobuilding_table.results[pandda_id])

        # Ligand RMSD: how good is autobuilding
        printer.pprint("# Analysing ligand rmsds")
        ligand_rmsds = AutobuildRMSDTable.from_directory(pandda_dir)
        rsmd_non_zero_mask = {dtag: rmsd for dtag, rmsd in ligand_rmsds.rmsds.items() if rmsd > 0.0}
        rmsd_small_mask = {dtag: rmsd for dtag, rmsd in ligand_rmsds.rmsds.items() if rmsd > 0.0}
        printer.pprint(f"rsmd_non_zero_mask: {len(rsmd_non_zero_mask)}")
        printer.pprint(
            f"Number of built events / number of events: {len(rsmd_non_zero_mask) / len(ligand_rmsds.rmsds)}"
        )
        printer.pprint(ligand_rmsds.table)
        # logs.LOG[pandda_id]["ligand_rmsds"] = ligand_rmsds

        # P(RMSD < 2.5 and RMSD > 0 | RSCC >0.7)
        selected_keys = set(rscc_large_mask.keys()).intersection(set(rscc_large_mask.keys())).intersection(set(rsmd_non_zero_mask.keys()))
        all_keys = set(rscc_large_mask.keys())
        printer.pprint(
            f"Number of built events with 0<rmsd<2.5 and rscc >0.7 / number of builds with rscc > 0.7: {len(selected_keys) / len(all_keys)}"
        )


        # Ranking
        printer.pprint("# Analysing ranking")
        # references = ReferenceStructures.from_dir(pandda_dir)
        # naive_ranking = PanDDARanking.from_pandda_dir(pandda_dir)
        # autobuilding_ranking = PanDDARanking.from_autobuild_rscc(autobuilding_table[pandda_id])
        # naive_enritchment = Enritchment.from_ranking(naive_ranking,
        #                                              references,
        #                                              )
        # autobuilding_enritchment = Enritchment.from_ranking(autobuilding_ranking,
        #                                                 references,
        #                                                 )
        # logs.LOG[pandda_id]["naive"] = naive_enritchment.enritchment
        # logs.LOG[pandda_id]["autobuilding"] = autobuilding_enritchment.enritchment
        #
        # printer.pprint(logs.LOG.dict)
        # except Exception as e:
        #     printer.pprint(e)

        # except Exception as e:
        #     print("# EXCEPTION: {}".format(e))

    to_json(results,
            config.analyse_file,
            )


if __name__ == "__main__":
    main()
