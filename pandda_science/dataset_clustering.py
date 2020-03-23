from pathlib import Path

import pandas as pd


def get_global_cluster_distribution_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))

    return parallel_pandda_df


def num_clusters_distribution_histogram():
    raise NotImplementedError()
