from typing import Dict, Tuple, NamedTuple
import os
import re
import shutil

import argparse
from pathlib import Path

import pandas as pd


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
    config = Config(event_table_path=Path(args.event_table_path),
                    autobuild_table_path=Path(args.autobuild_table_path),
                    out_dir_path=Path(args.out_dir_path),
                    )

    return config


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


def get_system_labels_true(model_table):
    systems = []
    for idx, row in model_table.iterrows():
        path = Path(row["pandda"]) / "processed_datasets"
        model_paths = [p.name
                       for p
                       in path.glob("*")]

        regex = "([a-z0-9]+)-x[0-9]+"

        m = re.search(regex,
                      model_paths[0],
                      )
        print(m)
        systems.append(m.group(0))

    return pd.Series(systems)


def get_system_labels_autobuilt(model_table,
                                event_table,
                                ):
    systems = []
    for idx, row in model_table.iterrows():
        event_table_row = event_table.loc[(row["dtag"], row["event_idx"])]
        path = Path(event_table_row["pandda"]) / "processed_datasets"
        model_paths = [p.name
                       for p
                       in path.glob("*")]

        regex = "([a-z0-9]+)-x[0-9]+"

        m = re.search(regex,
                      model_paths[0],
                      )
        systems.append(m.group(0))

    return pd.Series(systems)


def get_true_models_df(path: Path):
    true_model_table = pd.read_csv(str(path))
    true_model_table.set_index(["dtag", "event_idx"], inplace=True)
    print(true_model_table.head())

    true_model_table["system"] = get_system_labels_true(true_model_table)

    return true_model_table


def get_autobuilt_models_df(path: Path,
                            event_table,
                            ):
    true_model_table = pd.read_csv(str(path))
    true_model_table.set_index(["dtag", "event_idx"], inplace=True)
    print(true_model_table.head())

    true_model_table["system"] = get_system_labels_autobuilt(true_model_table)

    return true_model_table


def get_event_df(path: Path):
    return pd.read_csv(str(path))


def partition_by_system(true_model_df,
                        autobuilt_models_df,
                        test=0.2):
    systems = list(true_model_df["system"].unique())

    proportion = 0
    current_systems = []
    num_true_models = len(true_model_df)

    while proportion < 0.2:
        current_systems.append(systems.pop())
        num_datasets = sum([len(true_model_df[true_model_df["system"] == current_system])
                            for current_system
                            in current_systems
                            ])

        proportion = num_datasets / num_true_models

    return systems, current_systems


def make_table(true_model_df,
               autobuilt_models_df,
               event_table,
               systems,
               ):
    train_true = {system: true_model_df[true_model_df["system"] == system]
                  for system
                  in systems}
    train_autobuilt = {system: autobuilt_models_df[autobuilt_models_df["system"] == system]
                       for system
                       in systems}
    # [actually_built	analysed_resolution	data_path	dtag	event_idx	event_map_path	final_model_path	high_resolution	initial_model_path	interesting	ligand_confidence	ligand_placed	occupancy	pandda	viewed	x	y	z]
    true_table = pd.concat(train_true.values())
    # distance_to_event	dtag	event_idx	mean_rmsd	method	min_rmsd	num_candidates	time
    autobuilt_table = pd.concat(train_autobuilt.values())

    train_records = []
    for idx, row in true_table.iterrows():
        record = {"system": row["system"],
                  "dtag": row["dtag"],
                  "event_idx": row["event_idx"],
                  "resolution": row["analysed_resolution"],
                  "ligand_build_path": row[""],
                  "stripped_receptor_path": row[""],
                  "x": row["x"],
                  "y": row["y"],
                  "z": row["z"],
                  "data_path": row["event_map_path"],
                  "human_build": row["actually_built"],
                  "rmsd": 0,
                  }
        train_records.append(record)

    for idx, row in autobuilt_table.iterrows():
        event_record = event_table.loc[(row["dtag"], row["event_idx"])]
        record = {"system": row["system"],
                  "dtag": row["dtag"],
                  "event_idx": row["event_idx"],
                  "resolution": event_record["analysed_resolution"],
                  "ligand_build_path": autobuilt_table["autobuild_path"],
                  "stripped_receptor_path": autobuilt_table["stripped_receptor_path"],
                  "x": event_record["x"],
                  "y": event_record["y"],
                  "z": event_record["z"],
                  "data_path": row["data_path"],
                  "human_build": event_record["actually_built"],
                  "rmsd": row["min_rmsd"],
                  }
        train_records.append(record)

    return pd.DataFrame(train_records)


def output_table(table,
                 path,
                 ):
    table.to_csv(str(path))


if __name__ == "__main__":
    args = parse_args()

    config: Config = get_config(args)

    output: Output = setup_output(config.out_dir_path)

    true_model_df = get_true_models_df(config.event_table_path)

    autobuilt_models_df = get_autobuilt_models_df(config.autobuild_table_path)

    event_table = get_event_df(config.event_table_path)

    train_systems, test_systems = partition_by_system(true_model_df,
                                                      autobuilt_models_df,
                                                      )

    # system,dtag,event_idx,resolution,ligand_build_path,stripped_receptor_path,x,y,z,data_path,human_build,rmsd
    train_table = make_table(true_model_df,
                             autobuilt_models_df,
                             event_table,
                             train_systems,
                             )
    test_table = make_table(true_model_df,
                            autobuilt_models_df,
                            event_table,
                            test_systems,
                            )

    output_table(train_table,
                 output.train_table_path,
                 )
    output_table(test_table,
                 output.test_table_path,
                 )
