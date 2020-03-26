from typing import NamedTuple, Dict, List
import os
import shutil
import argparse
from pathlib import Path

import numpy as np

from pandda_types.data import Event


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--panddas_table",
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
    our_dir_path: Path
    events_df_path: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir_path),
                    root_path=Path(args.root_path),
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
    idx: int


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


class Comparison:
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


def get_precission(comparison):
    precission_vector = []



def get_recall(comparison: Comparison,
               cutoff: float = 5.0):
    recall_vector = []
    for event_id, event in comparison.new_pandda.events.items():
        comparison_event: Event = comparison.event_mapping.event_mapping_new_to_original[event_id]

        if comparison_event.actually_built:
            distance = get_distance(event, comparison_event)
            if distance < cutoff:
                recall_vector.append(1)
            else:
                recall_vector.append(0)

    recall = sum(recall_vector) / len(recall_vector)

    return recall


def Record:
    precission: float
    recall: float


if __name__ == "__main__":
    args = parse_args()

    config = get_config(args)

    output: Output = setup_output_directory(config.out_dir_path)

    pandda_table = get_pandda_table()

    event_table = get_event_table()

    match_panddas(original_panddas,
                  new_panddas,
                  )

    comparisons = []
    for original_pandda, new_pandda in pandda_matches:
        event_mapping = EventMapping(original_pandda, new_pandda)
        comparison = Comparison(original_pandda,
                                new_pandda,
                                event_mapping,
                                )

        comparisons.append(comparisons)

    comparison_df = make_comparison_df()
    output_comparison_df(comparison_df)
