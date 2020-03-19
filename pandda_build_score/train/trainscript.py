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
    parser.add_argument("-i", "--input_training_table",
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
    input_training_table_path: Path
    out_dir_path: Path


def get_training_config(args):
    config = Config(input_training_table_path=Path(args.input_training_table),
                    out_dir_path=Path(args.out_dir),
                    )

    return config


def output_trained_network(network: nn.Module, out_path: Path):
    pass


class Output:
    def __init__(self, path: Path):
        self.output_dir = path
        self.output_build_score_training_df_path = path / "output_build_score_training_df.csv"
        self.output_build_score_training_df_path = path / "output_build_score_test_df.csv"

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

    config: Config = get_training_config(args)

    output: Output = setup_output(config.out_dir_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader: DataLoader = get_train_dataloader(config.input_training_table_path)
    test_dataloader: DataLoader = get_test_dataloader(config.input_training_table_path)

    network: nn.Module = get_network()

    trained_network, build_score_training_df = train(network,
                                                     train_dataloader,
                                                     )

    build_score_test_df = test(network,
                               test_dataloader,
                               )

    output_build_score_training_df(build_score_training_df)
    output_build_score_test_df(build_score_test_df)
    output_trained_network(trained_network,
                           config.out_dir_path / "trained_network.pt",
                           )
