from __future__ import annotations

from autobuilding_paper.autobuild_types import *


def mask_xmap(xmap: Xmap, event: Event) -> Xmap:
    pass


def get_events(fs: PanDDAFilesystemModel, pandda_version: int) -> Dict[EventID, Event]:
    # Get event table
    pandda_analyse_events_file = fs.pandda_analyse_events_path
    event_table = pd.read_csv(str(pandda_analyse_events_file))

    # Get events
    events: Dict[EventID, Event] = {}
    for index, row in event_table.iterrows():
        dtag = Dtag(row["dtag"])
        event_idx = EventIDX(row["event_idx"])
        event_id = EventID(dtag, event_idx)

        occupancy = row["1-BDC"]

        centroid = (float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    )

        event_dir = fs.pandda_processed_datasets_dir / f"{dtag.dtag}"

        event: Event = Event(event_id=event_id,
                             centroid=centroid,
                             event_dir=event_dir,
                             )
        events[event_id] = event

    return events

def merge_mtzs(initial_mtz: Reflections, event_map_mtz: Reflections) -> Reflections:
    ...
