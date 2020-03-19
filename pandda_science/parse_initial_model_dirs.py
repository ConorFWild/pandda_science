from typing import Dict, Tuple, NamedTuple, List

import argparse
from pathlib import Path

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--root_dir",
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
    root_dir: Path
    out_dir: Path


def get_config(args):
    config = Config(root_dir=Path(args.root_dir),
                    out_dir=Path(args.out_dir),
                    )

    return config


def get_model_dirs(root_dir: Path):
    # regex = "**/pandda_inspect_events.csv"
    regex_initial_model = "*/*/processing/analysis/initial_model"
    regex_model_building = "*/*/processing/analysis/model_building"

    paths_initial_model = root_dir.glob(regex_initial_model)
    paths_model_building = root_dir.glob(regex_model_building)

    model_dirs: List[Path] = []

    for model_dir_path in paths_initial_model:
        model_dirs.append(model_dir_path)

    for model_dir_path in paths_model_building:
        model_dirs.append(model_dir_path)

    return model_dirs


def get_num_models(path: Path):
    paths = path.glob("*")
    paths_list = list(paths)
    return len(paths_list)


def get_model_dir_table(model_dirs: List[Path]):
    records = []

    for path in model_dirs:
        record = {"model_dir": path,
                  "num_models": get_num_models(path),
                  }

        records.append(record)

    df: pd.DataFrame = pd.DataFrame(records)

    return df


def output_table(df: pd.DataFrame, output_path: Path):
    df.to_csv(str(output_path))


if __name__ == "__main__":
    args = get_args()

    config: Config = get_config(args)

    print("Looking for pandda inspect tables in: {}".format(config.root_dir))
    model_dirs: List[Path] = get_model_dirs(config.root_dir)
    print("\tFound: {} model directories".format(len(model_dirs)))

    print("Constructing pandda table...")
    model_dir_table: pd.DataFrame = get_model_dir_table(model_dirs)

    print("Outputting pandda table to: {}".format(config.out_dir / "model_dirs.csv"))
    output_table(model_dir_table,
                 config.out_dir / "model_dirs.csv",
                 )
