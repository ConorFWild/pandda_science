from typing import Dict, Tuple, NamedTuple
import os
import shutil

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--event_table",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-r", "--rscc_table",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-o", "--out_dir_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    parser.add_argument("-n", "--name",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    args = parser.parse_args()

    return args


class Config(NamedTuple):
    event_table: Path
    rscc_table_path: Path
    out_dir_path: Path
    name: str


def get_training_config(args):
    config = Config(event_table=Path(args.event_table),
                    out_dir_path=Path(args.out_dir_path),
                    name=str(args.name),
                    rscc_table_path=Path(args.rscc_table),
                    )

    return config


class Output:
    def __init__(self, path: Path, name: str):
        self.output_dir = path
        self.output_table_path = path / "{}.csv".format(name)

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
            self.attempt_remove(self.output_dir)

        # Make output dirs
        self.attempt_mkdir(self.output_dir)


def setup_output(path: Path, overwrite: bool = False):
    output: Output = Output(path)
    output.make(overwrite=overwrite)
    return output


def output_table(table, path):
    table.to_csv(str(path))




def open_dataset(dataset):

    coot_script_path = write_coot_script()

    shell_command = make_shell_command(coot_script_path)

    process = open_coot(shell_command)

    return process

if __name__ == "__main__":
    args = parse_args()

    config: Config = get_training_config(args)

    output: Output = setup_output(config.out_dir_path)

    events = get_events(config.event_table)

    rsccs = get_rsccs(config.rscc_table_path)

    while True:
        dataset = select_dataset(events, rsccs)

        process = open_dataset(dataset)

        response = prompt_response()

        close_process(process)

        update_table(table, response)

        write_table(table)
