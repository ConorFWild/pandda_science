from pathlib import Path

import pandas as pd


def get_parallel_pandda_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))

    return parallel_pandda_df


def get_parallel_pandda_statistics_df(parallel_pandda_df: pd.DataFrame):
    pass


def parallel_pandda_speed_scatter(parallel_pandda_df: pd.DataFrame):
    pass
