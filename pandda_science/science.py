from typing import NamedTuple
import os
import shutil
import argparse
from pathlib import Path

import pandas as pd
from pandda_science.dataset import get_events_df
from pandda_science.parallel_pandda import (get_parallel_pandda_df,
                                            get_parallel_pandda_statistics_df,
                                            parallel_pandda_speed_scatter,
                                            )
from pandda_science.dataset_clustering import (get_global_cluster_distribution_df,
                                               num_clusters_distribution_histogram,
                                               num_modelable_distribution_histogram,
                                               )
from pandda_science.autobuilding import (get_autobuilding_results_df,
                                         get_relative_median_rmsd_by_system_df,
                                         get_autobuilding_rmsd_distribution_graph,
                                         get_autobuilding_rscc_distribution_graph,
                                         get_relative_median_rmsd_by_system_graph,
                                         get_autobuilding_rmsd_distribution_stats_table,
                                         )
from pandda_science.build_score import (get_build_score_train_df,
                                        get_build_score_test_df,
                                        get_event_size_ranking_df,
                                        get_rscc_ranking_df,
                                        get_build_score_ranking_df,
                                        )


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-o", "--out_dir_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )
    parser.add_argument("-e", "--events_df_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )
    parser.add_argument("-p", "--parallel_pandda_df_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )
    parser.add_argument("-c", "--dataset_clustering_df_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )
    parser.add_argument("-a", "--autobuilding_df_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )
    parser.add_argument("-tr", "--build_score_train_df_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )
    parser.add_argument("-ts", "--build_score_test_df_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    args = parser.parse_args()

    return args


class Config(NamedTuple):
    out_dir_path: Path
    events_df_path: Path
    parallel_pandda_df_path: Path
    dataset_clustering_df_path: Path
    autobuilding_df_path: Path
    build_score_train_df_path: Path
    build_score_test_df_path: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir_path),
                    events_df_path=Path(args.events_df_path),
                    parallel_pandda_df_path=Path(args.parallel_pandda_df_path),
                    dataset_clustering_df_path=Path(args.dataset_clustering_df_path),
                    autobuilding_df_path=Path(args.autobuilding_df_path),
                    build_score_train_df_path=Path(args.build_score_train_df_path),
                    build_score_test_df_path=Path(args.build_score_test_df_path),
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

        # parallel pandda
        self.parallel_pandda_statistics_table_path = self.parallel_pandda_out_dir_path / "statistics.csv"
        self.parallel_pandda_speed_graph_path = self.parallel_pandda_out_dir_path / "speed.png"

        # Dataset Clustering
        self.dataset_clustering_num_clusters_distribution_histogram_path = self.dataset_clustering_out_dir_path / "distribution_histogram.png"

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
        self.attempt_mkdir(self.dataset_out_dir_path)
        self.attempt_mkdir(self.parallel_pandda_out_dir_path)
        self.attempt_mkdir(self.dataset_clustering_out_dir_path)
        self.attempt_mkdir(self.autobuilding_out_dir_path)
        self.attempt_mkdir(self.build_score_out_dir_path)


def setup_output_directory(path: Path, overwrite: bool = False):
    output: Output = Output(path)
    output.make(overwrite)
    return output


if __name__ == "__main__":
    args = parse_args()

    config: Config = get_config(args)

    output: Output = setup_output_directory(config.out_dir_path)

    # Dataset
    events_df: pd.DataFrame = get_events_df(config.events_df_path)

    # Parallel PanDDA
    if config.parallel_pandda_df_path.is_file():
        parallel_pandda_df: pd.DataFrame = get_parallel_pandda_df(config.parallel_pandda_df_path)
        parallel_pandda_statistics_df: pd.DataFrame = get_parallel_pandda_statistics_df(parallel_pandda_df)
        parallel_pandda_speed_scatter(parallel_pandda_df)

    # Dataset Clustering
    if config.dataset_clustering_df_path.is_file():
        global_cluster_distribution_df: pd.DataFrame = get_global_cluster_distribution_df(
            config.dataset_clustering_df_path)
        num_clusters_distribution_histogram(global_cluster_distribution_df,
                                            config.out_dir_path)
        num_modelable_distribution_histogram(global_cluster_distribution_df,
                                             config.out_dir_path)

    # Autobuilding
    if config.autobuilding_df_path.is_file():
        autobuilding_results_df: pd.DataFrame = get_autobuilding_results_df(config.autobuilding_df_path)
        relative_median_rmsd_by_system_df = get_relative_median_rmsd_by_system_df(autobuilding_results_df)
        get_autobuilding_rmsd_distribution_stats_table(relative_median_rmsd_by_system_df,
                                                       config.out_dir_path / "autobuilding_stats.csv",
                                                       cutoffs=[2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0],
                                                       )
        get_autobuilding_rmsd_distribution_graph(relative_median_rmsd_by_system_df,
                                                 config.out_dir_path,
                                                 cutoffs=[2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0],
                                                 )
        get_autobuilding_rscc_distribution_graph(autobuilding_results_df)
        get_relative_median_rmsd_by_system_graph(relative_median_rmsd_by_system_df)

    # Build Quality classification
    if config.build_score_train_df_path.is_file() and config.build_score_test_df_path.is_file():
        build_score_train_df: pd.DataFrame = get_build_score_train_df()
        build_score_test_df: pd.DataFrame = get_build_score_test_df()

        event_size_ranking_df: pd.DataFrame = get_event_size_ranking_df()
        autobuilding_rscc_ranking_df: pd.DataFrame = get_rscc_ranking_df()
        build_score_ranking_df: pd.DataFrame = get_build_score_ranking_df()
