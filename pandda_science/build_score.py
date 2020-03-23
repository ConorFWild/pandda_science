from pathlib import Path

import pandas as pd


def get_build_score_train_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))

    return parallel_pandda_df


def get_build_score_test_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))

    return parallel_pandda_df


def get_event_size_ranking_df():
    raise NotImplementedError()


def get_rscc_ranking_df():
    raise NotImplementedError()


def get_build_score_ranking_df():
    raise NotImplementedError()
