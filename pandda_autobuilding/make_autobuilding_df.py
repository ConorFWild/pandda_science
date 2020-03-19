from typing import NamedTuple
import os
import shutil
import argparse
from pathlib import Path


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
    our_dir_path: Path
    events_df_path: Path


def get_config(args):
    config = Config(our_dir_path=Path(args.out_dir_path),
                    events_df_path=Path(args.events_df_path),
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path
        self.dataset_out_dir_path = out_dir_path / "dataset"
        self.parallel_pandda_out_dir_path = out_dir_path / "parallel_pandda"
        self.dataset_clustering_out_dir_path = out_dir_path / "dataset_clustering"
        self.autobuilding_out_dir_path = out_dir_path / "autobuilding"
        self.build_score_out_dir_path = out_dir_path / "build_score"

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

    output: Output = setup_output_directory(config.our_dir_path)


