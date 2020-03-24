from typing import NamedTuple, List
import os
import shutil
import argparse
from pathlib import Path
import subprocess

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-c", "--dataset_clustering_table_path",
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
    out_dir_path: Path
    dataset_clustering_table_path: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir_path),
                    dataset_clustering_table_path=Path(args.model_dirs_table_path),
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path

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


def get_dataset_clustering_dfs(path: Path):
    pattern = "*/processed/labelled_embedding.csv"
    dataset_clustering_df_paths = path.glob(pattern)
    dataset_clustering_dfs = [pd.read_csv(str(p))
                              for p
                              in dataset_clustering_df_paths
                              ]

    return dataset_clustering_dfs


def get_num_modelable_clusters(dataset_clustering_df,
                               cutoff=60,
                               ):
    cluster_labels = dataset_clustering_df["cluster"].unique()
    modelable = 0
    for cluster_label in cluster_labels:
        if cluster_label == -1:
            continue
        cluster_df = dataset_clustering_df[dataset_clustering_df["cluster"] == cluster_label]
        if len(cluster_df) > cutoff:
            modelable = modelable + 1
    return modelable


def summarise_get_dataset_clustering_dfs(dataset_clustering_dfs: List[pd.DataFrame]):
    records = []
    for dataset_clustering_df in dataset_clustering_dfs:
        cluster_labels = dataset_clustering_df["cluster"].unique()

        num_clusters = len(cluster_labels[cluster_labels != -1])
        num_modelable_clusters = get_num_modelable_clusters(dataset_clustering_df)

        record = {"num_clusters": num_clusters,
                  "num_modelable_clusters": num_modelable_clusters,
                  }
        records.append(record)
    return pd.DataFrame(records)


def output_summary_df(summary_df,
                      output_path,
                      ):
    summary_df.to_csv(str(output_path))


if __name__ == "__main__":
    args = parse_args()

    config = get_config(args)

    output: Output = setup_output_directory(config.out_dir_path)

    dataset_clustering_dfs = get_dataset_clustering_dfs(config.dataset_clustering_table_path)

    summary_df = summarise_get_dataset_clustering_dfs(dataset_clustering_dfs)

    output_summary_df(summary_df,
                      output.out_dir_path / "dataset_clustering_summary.csv",
                      )
