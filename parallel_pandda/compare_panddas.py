from typing import NamedTuple, Dict, List
import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pandda_types.data import Event


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--events_df_path",
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
    events_df_path: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir_path),
                    events_df_path=Path(args.events_df_path),
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path
        self.pandda_comparison_table_path = out_dir_path / "pandda_comparison_table.csv"

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


class EventID:
    dtag: str
    event_idx: int

    def __init__(self, dtag, event_idx):
        self.dtag = dtag
        self.event_idx = event_idx

    def __hash__(self):
        return hash((self.dtag, self.event_idx))

    def __eq__(self, other):
        return (self.dtag, self.event_idx) == (other.dtag, other.event_idx)


class PanDDA:
    data_dir: Path
    pandda_dir: Path
    events_table_path: Path
    events: Dict[EventID, Event]


class EventMapping:
    event_mapping_original_to_new: Dict[EventID, EventID]
    event_mapping_new_to_original: Dict[EventID, EventID]

    def __init__(self, original_pandda: PanDDA, new_pandda: PanDDA):
        self.event_mapping_original_to_new = self.map_events(original_pandda,
                                                             new_pandda,
                                                             )
        self.event_mapping_new_to_original = self.map_events(new_pandda,
                                                             original_pandda,
                                                             )

    def map_events(self, pandda_from, pandda_to):
        mapping = {}
        for event_id, event in pandda_from.events.items():
            closest_event_id = self.get_closest_event_id(event,
                                                         pandda_to.events,
                                                         )
            mapping[event_id] = closest_event_id

        return mapping

    def get_closest_event_id(self, event, events_dict):
        distances = {}
        for event_id, comparison_event in events_dict.items():
            distance = self.get_distance(event, comparison_event)
            distances[event_id] = distance

        min_distance_index = np.argmin(list(distances.values()))
        min_distance_event_id = list(distances.keys())[min_distance_index]

        return min_distance_event_id

    def get_distance(self, event1, event2):
        event1_coord = np.array([event1.x, event1.y, event1.z])
        event2_coord = np.array([event2.x, event2.y, event2.z])

        distance_vector = event2_coord - event1_coord

        distance = np.linalg.norm(distance_vector)

        return distance


def get_distance(event1, event2):
    event1_coord = np.array([event1.x, event1.y, event1.z])
    event2_coord = np.array([event2.x, event2.y, event2.z])

    distance_vector = event2_coord - event1_coord

    distance = np.linalg.norm(distance_vector)

    return distance


class ComparisonSet:
    original_pandda: PanDDA
    new_pandda: PanDDA
    event_mapping: EventMapping

    def __init__(self, original_pandda,
                 new_pandda,
                 event_mapping,
                 ):
        self.original_pandda = original_pandda
        self.new_pandda = new_pandda
        self.event_mapping = event_mapping


def get_precission(comparison: ComparisonSet,
                   cutoff_distance_to_model: float = 8.0,
                   cutoff_distance_to_event: float = 8.0,
                   ):
    precission_vector = []
    for event_id, event in comparison.new_pandda.events.items():
        comparison_event_id: EventID = comparison.event_mapping.event_mapping_new_to_original[event_id]
        comparison_event = comparison.original_pandda.events[comparison_event_id]

        if comparison_event.actually_built:
            if (comparison_event.distance_to_ligand_model > 0) and (
                    comparison_event.distance_to_ligand_model < cutoff_distance_to_model):
                distance = get_distance(event, comparison_event)
                if distance < cutoff_distance_to_event:
                    precission_vector.append(1)

    precission = sum(precission_vector) / len(comparison.new_pandda.events)

    return precission


def get_recall(comparison: ComparisonSet,
               cutoff_distance_to_model: float = 8.0,
               cutoff_distance_to_event: float = 8.0,
               ):
    recall_vector = []
    for event_id, event in comparison.new_pandda.events.items():
        comparison_event_id: EventID = comparison.event_mapping.event_mapping_new_to_original[event_id]
        comparison_event = comparison.original_pandda.events[comparison_event_id]

        if comparison_event.actually_built:
            if (comparison_event.distance_to_ligand_model > 0) and (
                    comparison_event.distance_to_ligand_model < cutoff_distance_to_model):
                distance = get_distance(event, comparison_event)
                if distance < cutoff_distance_to_event:
                    recall_vector.append(1)
                else:
                    recall_vector.append(0)

    recall = sum(recall_vector) / len(recall_vector)

    return recall


def get_precission_base(pandda: PanDDA,
                        cutoff_distance_to_model: float = 8.0):
    num_events = len(pandda.events)
    hits = []
    for event_id, event in pandda.events.items():
        if event.actually_built:
            if event.distance_to_ligand_model < cutoff_distance_to_model:
                hits.append(1)

    precission = sum(hits) / num_events

    return precission


class ComparisonRecord:
    original_pandda_precission: float
    new_pandda_precission: float
    new_pandda_recall: float

    def __init__(self,
                 comaprison_set: ComparisonSet,
                 original_pandda_precission,
                 new_pandda_precission,
                 new_pandda_recall,
                 ):
        self.model_dir = comaprison_set.original_pandda.data_dir
        self.original_pandda_dir = comaprison_set.original_pandda.pandda_dir
        self.new_pandda_dir = comaprison_set.new_pandda.pandda_dir
        self.original_pandda_num_events = len(comaprison_set.original_pandda.events)
        self.new_pandda_num_events = len(comaprison_set.new_pandda.events)

        self.original_pandda_precission = original_pandda_precission
        self.new_pandda_precission = new_pandda_precission
        self.new_pandda_recall = new_pandda_recall

    def to_dict(self):
        record = {}
        record["model_dir"] = self.model_dir
        record["original_pandda_dir"] = self.original_pandda_dir
        record["new_pandda_dir"] = self.new_pandda_dir
        record["original_pandda_num_events"] = self.original_pandda_num_events
        record["new_pandda_num_events"] = self.new_pandda_num_events

        record["original_pandda_precission"] = self.original_pandda_precission
        record["new_pandda_precission"] = self.new_pandda_precission
        record["new_pandda_recall"] = self.new_pandda_recall
        return record


def make_comparison_table(comparison_sets: List[ComparisonSet]):
    comparison_records = []
    for comparison_set in comparison_sets:
        base_pandda_precission = get_precission_base(comparison_set.original_pandda)
        new_pandda_precission = get_precission(comparison_set)
        new_pandda_recall = get_recall(comparison_set)
        record = ComparisonRecord(comparison_set,
                                  base_pandda_precission,
                                  new_pandda_precission,
                                  new_pandda_recall,
                                  )
        comparison_records.append(record.to_dict())
    return pd.DataFrame(comparison_records)


def match_panddas(original_panddas: List[PanDDA],
                  new_panddas: List[PanDDA],
                  ):
    matches = []

    for original_pandda in original_panddas:
        for new_pandda in new_panddas:
            if new_pandda.data_dir == original_pandda.data_dir:
                matches.append((original_pandda, new_pandda))
                break

    return matches


def get_events(events_table: pd.DataFrame):
    events = {}
    for idx, row in events_table.iterrows():
        event = Event.from_record(row)
        event_id = EventID(row["dtag"], row["event_idx"])
        events[event_id] = event

    return events


def panddas_from_event_table(pandda_event_table):
    initial_dirs = pandda_event_table["model_dir"].unique()

    panddas = []

    for initial_dir in initial_dirs:
        initial_dir_table = pandda_event_table[pandda_event_table["model_dir"] == initial_dir]
        pandda_names = initial_dir_table["pandda_name"].unique()
        for pandda_name in pandda_names:
            pandda_name_table = initial_dir_table[initial_dir_table["pandda_name"] == pandda_name]
            events = get_events(pandda_name_table)
            pandda = PanDDA()
            panddas.append(pandda)

    return panddas


def get_panddas(event_table: pd.DataFrame):
    new_panddas_table = event_table[event_table["pandda_name"] == "test_pandda_parallel"]
    old_pandda_table = event_table[event_table["pandda_name"] != "test_pandda_parallel"]

    original_panddas = panddas_from_event_table(old_pandda_table)
    new_panddas = panddas_from_event_table(new_panddas_table)

    return original_panddas, new_panddas


def output_comparison_table(comparison_table,
                            output_path,
                            ):
    comparison_table.to_csv(str(output_path))


def get_event_table(event_table_path: Path):
    event_table = pd.DataFrame(str(event_table_path))
    return event_table


def main():
    args = parse_args()

    config = get_config(args)

    output: Output = setup_output_directory(config.out_dir_path)

    event_table = get_event_table(config.events_df_path)

    original_panddas, new_panddas = get_panddas(event_table)

    pandda_matches = match_panddas(original_panddas,
                                   new_panddas,
                                   )

    comparisons = []
    for original_pandda, new_pandda in pandda_matches:
        event_mapping = EventMapping(original_pandda, new_pandda)
        comparison = ComparisonSet(original_pandda,
                                   new_pandda,
                                   event_mapping,
                                   )

        comparisons.append(comparison)

    comparison_df = make_comparison_table(comparisons)
    output_comparison_table(comparison_df,
                            output.pandda_comparison_table_path,
                            )


if __name__ == "__main__":
    main()
