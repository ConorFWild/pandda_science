from typing import Dict, Tuple, NamedTuple
import os
import shutil

import argparse
from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from pandda_types.data import PanDDA, Event

from pandda_build_score.train.dataset import get_dataloader
from pandda_build_score.train.network import get_network
from pandda_build_score.train.training import train


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-e", "--event_table_path",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )
    parser.add_argument("-a", "--autobuild_table_path",
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

    return args


class Config(NamedTuple):
    event_table_path: Path
    autobuild_table_path: Path
    out_dir_path: Path


def get_config(args):
    config = Config(input_training_table_path=Path(args.input_training_table),
                    out_dir_path=Path(args.out_dir),
                    )

    return config


def output_trained_network(network: nn.Module, out_path: Path):
    pass


class Output:
    def __init__(self, path: Path):
        self.output_dir = path
        self.train_table_path = path / "train_table.csv"
        self.test_table_path = path / "test_table.csv"

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


if __name__ == "__main__":
    args = parse_args()

    config: Config = get_config(args)

    output: Output = setup_output(config.out_dir_path)

    true_model_df = get_true_models_df(config.event_table_path)

    autobuilt_models_df = get_autobuilt_models_df(config.autobuild_table_path)

    train_systems, test_systems = partition_by_system(true_model_df,
                                                      autobuilt_models_df,
                                                      )

    train_table = make_table(true_model_df,
                             autobuilt_models_df,
                             train_systems,
                             )
    test_table = make_table(true_model_df,
                            autobuilt_models_df,
                            test_systems,
                            )

    output_table(train_table,
                 output.train_table_path,
                 )
    output_table(test_table,
                 output.test_table_path,
                 )