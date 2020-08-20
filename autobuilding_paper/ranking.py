import dataclasses
import typing

import pandas as pd

from autobuilding_paper.constants import *
from autobuilding_paper.lib import EventID, Event


@dataclasses.dataclass()
class PanDDARanking:
    rankings: typing.List[EventID]

    @staticmethod
    def from_autobuild_rscc(autobuilds):
        sorted_list = sorted(autobuilds,
                             key=lambda x: autobuilds[x].rscc,
                             )
        return PanDDARanking(sorted_list)

    @staticmethod
    def from_pandda_dir(pandda_dir):
        analysed_events_file = pandda_dir / PANDDA_ANALYSES_DIR / PANDDA_ANALYSE_EVENTS_FILE

        table = pd.read_csv(str(analysed_events_file))

        events = []
        for index, series in table.iterrows():
            event = Event.from_dict(series)
            events.append(event)

        sorted_events = sorted(events,
                               key=lambda x: x.cluster_size,
                               )

        return sorted_events


@dataclasses.dataclass()
class Enritchment:
    enritchment: float

    @staticmethod
    def from_ranking(ranking, references, alpha=0.2):

        hits = 0
        for i, event_id in enumerate(ranking):
            if i > len(ranking) * alpha:
                break

            if event_id in references:
                hits = hits + 1

        enritchment = hits / (len(ranking) * alpha)

        return Enritchment(enritchment)

