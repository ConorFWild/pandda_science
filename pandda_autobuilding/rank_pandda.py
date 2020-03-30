from typing import NamedTuple, List, Dict
import os
import shutil
import re
import argparse
from pathlib import Path
import json
from functools import partial

import numpy as np
import pandas as pd

import joblib

from pandda_types.data import (Event, )
from pandda_autobuilding.result import Result
from pandda_autobuilding.model import (BioPandasModel, get_rmsd_dfs, )


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--pandda_dir",
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
    out_dir_path: Path
    events_df_path: Path
    autobuilding_dir_path: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir),
                    events_df_path=Path(args.events_df_path),
                    autobuilding_dir_path=Path(args.autobuilding_dir_path)
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path
        self.ranking_table_csv_path = out_dir_path / "ranking_table.csv"

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



if __name__ == "__main__":
    args = parse_args()

    config = get_config(args)

    output: Output = setup_output_directory(config.out_dir_path)

    print("Getting events from csv...")
    events = get_events(config.events_df_path)
    print("\tGot events csv with: {} events".format(len(events)))

