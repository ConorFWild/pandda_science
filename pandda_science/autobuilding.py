from pathlib import Path

import pandas as pd


def get_autobuilding_results_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))

    return parallel_pandda_df
