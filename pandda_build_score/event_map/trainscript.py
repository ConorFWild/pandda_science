from typing import Dict, Tuple, NamedTuple
import os
import shutil

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from pandda_build_score.event_map.preprocess import get_train_test_split
from pandda_build_score.event_map.dataset import get_dataloader
from pandda_build_score.event_map.network import get_network
from pandda_build_score.event_map.training import train
from pandda_build_score.event_map.testing import test


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--event_table",
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
    input_training_table_path: Path
    out_dir_path: Path
    shape: np.ndarray


def get_training_config(args):
    config = Config(input_training_table_path=Path(args.event_table),
                    out_dir_path=Path(args.out_dir_path),
                    shape=np.array([32, 32, 32]),
                    )

    return config


def output_trained_network(network, out_path: Path):
    pass


class Output:
    def __init__(self, path: Path):
        self.output_dir = path
        self.train_table_path = path / "train_table.csv"
        self.test_table_path = path / "test_table.csv"
        self.train_score_table_path = path / "train_score_table_path.csv"
        self.test_score_table_path = path / "test_score_table_path.csv"

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


if __name__ == "__main__":
    args = parse_args()

    config: Config = get_training_config(args)

    output: Output = setup_output(config.out_dir_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output.train_table_path.exists():
        train_table = pd.read_csv(str(output.train_table_path))
        test_table = pd.read_csv(str(output.test_table_path))
        train_dataloader = get_dataloader(train_table,
                                          shape=config.shape,
                                          )
        test_dataloader = get_dataloader(test_table,
                                         shape=config.shape,
                                         )

    else:
        event_table = pd.read_csv(config.input_training_table_path)
        train_table, test_table = get_train_test_split(event_table)  # TODO: Needs implementing
        print(len(train_table))
        print(len(test_table))
        print(len(train_table[train_table["actually_built"]==True]))
        print(len(test_table[test_table["actually_built"]==True]))
        # exit()
        train_table.to_csv(str(output.train_table_path))
        test_table.to_csv(str(output.test_table_path))
        train_dataloader = get_dataloader(train_table,
                                          shape=config.shape,
                                          )
        test_dataloader = get_dataloader(test_table,
                                         shape=config.shape,
                                         )

    network = get_network(config.shape)

    trained_network, train_score_table = train(network,
                                               train_dataloader,
                                               )

    test_score_table = test(network,
                            test_dataloader,
                            )

    output_table(train_score_table,
                 output.train_score_table_path,
                 )
    output_table(test_score_table,
                 output.test_score_table_path,
                 )
    output_trained_network(trained_network,
                           config.out_dir_path / "trained_network.pt",
                           )
