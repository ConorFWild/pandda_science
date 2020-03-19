from pathlib import Path

import numpy as np
import pandas as pd

from pandda_science.graphs import scatter


def get_parallel_pandda_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))

    return parallel_pandda_df


def get_successful_runs(pandda_df: pd.DataFrame):
    return len(pandda_df[pandda_df["success"] == 1])


def get_parallel_pandda_statistics_df(parallel_pandda_df: pd.DataFrame,
                                      output_path: Path,
                                      ):
    original_pandda_df = parallel_pandda_df[parallel_pandda_df["type"] == "original"]
    parallel_pandda_df = parallel_pandda_df[parallel_pandda_df["type"] == "parallel"]

    # Number of successful runs
    sucessful_runs_original = get_successful_runs(original_pandda_df)
    sucessful_runs_parallel = get_successful_runs(parallel_pandda_df)

    # Total number of events
    num_events_original = get_num_events(original_pandda_df)

    # Average event size

    # precission

    # Recall




def filter_both_ran(parallel_pandda_df: pd.DataFrame):
    # Algorithm: iterate over unique initial inputs and check if there are two successful runs
    return


def parallel_pandda_speed_scatter(parallel_pandda_df: pd.DataFrame,
                                  output_path: Path,
                                  ):
    parallel_pandda_df = filter_both_ran(parallel_pandda_df)
    original_pandda_records = parallel_pandda_df[parallel_pandda_df["type"] == "original"]
    parallel_pandda_records = parallel_pandda_df[parallel_pandda_df["type"] == "parallel"]
    original_runtime_array = np.array(original_pandda_records["runtime"])
    parallel_runtime_array = np.array(original_pandda_records["runtime"])
    scatter(original_runtime_array,
            parallel_runtime_array,
            output_path=output_path,
            )
