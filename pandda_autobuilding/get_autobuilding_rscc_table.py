from typing import NamedTuple, List
import os
import time
import subprocess
import shutil
import re
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import joblib

import luigi

from pandda_types.data import Event


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--event_table_path",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-a", "--autobuilding_dir",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    parser.add_argument("-o", "--out_dir_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    args = parser.parse_args()

    return args


class Config(NamedTuple):
    out_dir_path: Path
    event_table_path: Path
    autobuilding_dir: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir_path),
                    event_table_path=Path(args.event_table_path),
                    autobuilding_dir=Path(args.autobuilding_dir)
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path
        self.rscc_table_path = out_dir_path / "rscc.csv"

    def attempt_mkdir(self, path: Path):
        try:
            os.mkdir(str(path))
        except Exception as e:
            print(e)

    def attempt_remove(self, path: Path):
        try:
            shutil.rmtree(path,
                          ignore_errors=True,
                          )
        except Exception as e:
            print(e)

    def make(self, overwrite=False):
        # Overwrite old results as appropriate
        if overwrite is True:
            self.attempt_remove(self.out_dir_path)

        # Make output dirs
        self.attempt_mkdir(self.out_dir_path)


def setup_output_directory(path: Path, overwrite: bool = False):
    output: Output = Output(path)
    output.make(overwrite)
    return output


def get_event_table(path):
    events = []
    event_table = pd.read_csv(str(path))
    for idx, event_row in event_table.iterrows():
        # if event_row["actually_built"] is True:
        event = Event.from_record(event_row)
        events.append(event)
        # else:
        #     continue

    return events


class Build:
    def __init__(self, pandda_name, dtag, event_idx, rscc):
        self.pandda_name = pandda_name
        self.dtag = dtag
        self.event_idx = event_idx
        self.rscc = rscc

    def to_record(self):
        record = {}
        record["pandda_name"] = self.pandda_name
        record["dtag"] = self.dtag
        record["event_idx"] = self.event_idx
        record["rscc"] = self.rscc
        return record


def get_rscc_table(events, autobuilding_dir):
    records = []
    for event in events:
        event_autobuild_dir = autobuilding_dir / "{}_{}_{}".format(event.pandda_name,
                                                                   event.dtag,
                                                                   event.event_idx,
                                                                   )

        phenix_results_file = event_autobuild_dir / "phenix_event" / "LigandFit_run_1_" / "LigandFit_summary.dat"

        rscc_regex = "[\s]+1[\s]+[0-9\.]+[\s]+([0-9\.]+)"

        if not phenix_results_file.exists():
            print("\tCould not find results!")
            continue

        with open(str(phenix_results_file), "r") as f:
            result_string = f.read()

        match = re.findall(rscc_regex, result_string)

        print(result_string)
        print(match)

        rscc_string = match[0]
        rscc = float(rscc_string)

        record = {}
        record["pandda_name"] = event.pandda_name
        record["dtag"] = event.dtag
        record["event_idx"] = event.event_idx
        record["rscc"] = rscc

        records.append(record)

    return pd.DataFrame(records)


def main():
    print("Parsing args")
    args = parse_args()

    print("Geting Config...")
    config = get_config(args)

    print("Setiting up output...")
    output = setup_output_directory(config.out_dir_path)

    print("Getting event table...")
    events = get_event_table(config.event_table_path)
    print("\tGot {} events!".format(len(events)))

    rscc_table = get_rscc_table(events, config.autobuilding_dir)

    rscc_table.to_csv(str(output.rscc_table_path))


if __name__ == "__main__":
    main()
