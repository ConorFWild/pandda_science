from typing import List
from pathlib import Path

import pandas as pd


def get_panddas_df(path: Path):
    pass

def get_events_df(path: Path):
    events_df: pd.DataFrame = pd.read_csv(str(path))
    return events_df

