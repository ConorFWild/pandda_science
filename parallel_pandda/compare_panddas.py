from typing import NamedTuple, Dict, List
import os
import shutil
import argparse
import json
import cloudpickle
from pathlib import Path

import numpy as np
import pandas as pd

from pandda_types.data import Event


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-ie", "--events_df_path",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-in", "--new_panddas_dir",
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
    new_panddas_dir: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir_path),
                    events_df_path=Path(args.events_df_path),
                    new_panddas_dir=Path(args.new_panddas_dir),
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path
        self.pandda_comparison_table_path = out_dir_path / "pandda_comparison_table.csv"
        self.comparison_json_path = out_dir_path / "comparison.json"

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

    def to_tuple(self):
        return (self.dtag, self.event_idx)


class PanDDA:
    data_dir: Path
    pandda_dir: Path
    events_table_path: Path
    events: Dict[EventID, Event]

    def __init__(self,
                 data_dir,
                 pandda_dir,
                 events_table_path,
                 events,
                 ):
        self.data_dir = data_dir
        self.pandda_dir = pandda_dir
        self.events_table_path = events_table_path
        self.events = events

    @staticmethod
    def from_event_table(event_table: pd.DataFrame):
        raise NotImplementedError()

    @staticmethod
    def from_pandda_path(path: Path):
        pandda_dir = path
        events_csv_path = path / "analyses" / "pandda_analyse_events.csv"

        pandda_json_path = path / "pandda.json"

        with open(str(pandda_json_path), "r") as f:
            pandda_json_string = f.read()
            pandda_json_dict = json.loads(pandda_json_string)

        data_dirs = Path(pandda_json_dict["data_dirs"]).parent

        event_table = pd.read_csv(str(events_csv_path))

        events = Event.from_pandda_path(pandda_dir,
                                        data_dirs,
                                        )

        pandda = PanDDA(data_dir=data_dirs,
                        pandda_dir=pandda_dir,
                        events_table_path=events_csv_path,
                        events=events,
                        )
        return pandda


class EventMapping:
    event_mapping_original_to_new: Dict[EventID, EventID]
    event_mapping_new_to_original: Dict[EventID, EventID]

    def __init__(self, original_pandda: PanDDA = None,
                 new_pandda: PanDDA = None,
                 mappings_dict=None, ):
        if mappings_dict is not None:

            self.event_mapping_original_to_new = {(key[0], key[1]): (value[0], value[1])
                                                  for key, value
                                                  in mappings_dict["event_mapping_original_to_new"]
                                                  }
            self.event_mapping_new_to_original = {(key[0], key[1]): (value[0], value[1])
                                                  for key, value
                                                  in mappings_dict["event_mapping_new_to_original"]
                                                  }
        else:
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

    def to_dict(self):
        mapping_dict = {}
        mapping_dict["event_mapping_original_to_new"] = {key: value
                                                         for key, value
                                                         in self.event_mapping_original_to_new.items()
                                                         }
        mapping_dict["event_mapping_new_to_original"] = {key: value
                                                         for key, value
                                                         in self.event_mapping_new_to_original.items()
                                                         }
        return mapping_dict


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
    # num_true_positives = len([e for e in comparison.original_pandda.events.values() if e.actually_built])
    for event_id, event in comparison.new_pandda.events.items():
        comparison_event_id: EventID = comparison.event_mapping.event_mapping_new_to_original[event_id]
        comparison_event = comparison.original_pandda.events[comparison_event_id]

        # if comparison_event.actually_built:
        #     pass



        # print("comparison_event.distance_to_ligand_model: {}".format(comparison_event.distance_to_ligand_model))
        if comparison_event.distance_to_ligand_model > 0:
            if comparison_event.distance_to_ligand_model < cutoff_distance_to_model:
                distance = get_distance(event, comparison_event)
                if distance < cutoff_distance_to_event:
                    recall_vector.append(1)
                else:
                    recall_vector.append(0)
            else:
                # Original event was not acutall a hit
                pass

        else:
            # Comparison event could not get a distance to model
            pass

    if len(recall_vector) == 0:
        return -1

    recall = sum(recall_vector) / len(recall_vector)

    return recall


def get_precission_base(pandda: PanDDA,
                        cutoff_distance_to_model: float = 8.0):
    num_events = len(pandda.events)
    if num_events == 0:
        print("\t\t\tpandda {} has 0 events!".format(pandda.pandda_dir))
        return 0

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


def make_comparison_table(comparison_sets: Dict):
    comparison_records = []
    for key, comparison_set in comparison_sets.items():
        print("\tProcessing match: {}".format(key))
        print("\tOld pandda has {} events".format(len(comparison_set.original_pandda.events)))
        print("\tNew panddas has {} events".format(len(comparison_set.new_pandda.events)))
        print("\tOld pandda has {} built events".format(len([event
                                                             for event
                                                             in comparison_set.original_pandda.events.values()
                                                             if event.actually_built])))

        if len(comparison_set.original_pandda.events) == 0:
            print("\t\tPanDDA {} has no events: cannot calculate precision: skipping!".format(comparison_set.original_pandda.pandda_dir))
            continue
        base_pandda_precission = get_precission_base(comparison_set.original_pandda)
        print("\t\tOld pandda precission: {}".format(base_pandda_precission))

        if len(comparison_set.new_pandda.events) == 0:
            print("\t\tNew PanDDA {} has no events: cannot calculate precision: skipping!").format(comparison_set.new_pandda.pandda_dir)
            continue
        new_pandda_precission = get_precission(comparison_set)
        print("\t\tNew PanDDA precision: {}".format(new_pandda_precission))

        if len([event for event in comparison_set.original_pandda.events.values() if event.actually_built]) == 0:
            print("\t\tOld PanDDA {} has no built events: cannot calculate recall: skipping!".format(comparison_set.original_pandda.pandda_dir))
            continue
        if len([event
                for event
                in comparison_set.original_pandda.events.values()
                if (event.distance_to_ligand_model > 0) and (event.distance_to_ligand_model <8)]) ==0:
            print("\t\tOld PanDDA {} has no built events with a known distance to event: cannot calculate recall: skipping!".format(
                comparison_set.original_pandda.pandda_dir))
            continue

        new_pandda_recall = get_recall(comparison_set)
        if new_pandda_recall == -1:
            print("\t\tCould not compare! Skipping!")
            continue
        print("\t\tNew PanDDA recall: {}".format(new_pandda_recall))

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
            if str(new_pandda.data_dir) == str(original_pandda.data_dir):
                matches.append((original_pandda, new_pandda))

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
            pandda_dir = Path(initial_dir_table["pandda"].iloc[0])
            events_table_path = pandda_dir / "analyses" / "pandda_analyse_events.csv"

            pandda = PanDDA(initial_dir,
                            pandda_dir,
                            events_table_path,
                            events,
                            )
            panddas.append(pandda)

    return panddas


def get_old_panddas(event_table: pd.DataFrame):
    # new_panddas_table = event_table[event_table["pandda_name"] == "test_pandda_parallel"]
    old_pandda_table = event_table[event_table["pandda_name"] != "test_pandda_parallel"]

    original_panddas = panddas_from_event_table(old_pandda_table)
    # new_panddas = panddas_from_event_table(new_panddas_table)

    return original_panddas


def get_new_panddas(path):
    pandda_paths = path.glob("*")

    panddas = []

    for pandda_path in pandda_paths:
        events_csv_path = pandda_path / "analyses" / "pandda_analyse_events.csv"
        if not events_csv_path.exists():
            print("\tCouldn't find an analsy events path at: {}".format(events_csv_path))
            continue
        pandda = PanDDA.from_pandda_path(pandda_path)
        panddas.append(pandda)

    return panddas


def output_comparison_table(comparison_table,
                            output_path,
                            ):
    comparison_table.to_csv(str(output_path))


def get_event_table(event_table_path: Path):
    event_table = pd.read_csv(str(event_table_path))
    return event_table


def to_json(comparisons: Dict,
            output_path: Path,
            ):
    comparison_dict = {}
    for key, comparison in comparisons.items():
        comparison_dict[key] = comparison.event_mapping.to_dict()

    json_string = json.dumps(comparison_dict)

    with open(str(output_path), "w") as f:
        f.write(json_string)


def from_json(matches, json_path):
    comparsons = {}

    with open(str(json_path), "r") as f:
        json_string = f.read()

    comparisons_dict = json.loads(json_string)

    for original_pandda, new_pandda in matches:
        mapping_dict = comparisons_dict[(original_pandda.pandda_dir, new_pandda.pandda_dir)]
        event_mapping = EventMapping(mappings_dict=mapping_dict)

        comparison = ComparisonSet(original_pandda,
                                   new_pandda,
                                   event_mapping,
                                   )

        comparsons[(original_pandda.pandda_dir, new_pandda.pandda_dir)] = comparison

    return comparsons


def main():
    print("Parsing args...")
    args = parse_args()

    print("Configuring...")
    config = get_config(args)

    print("Setting up output...")
    output: Output = setup_output_directory(config.out_dir_path)

    print("Getting event table...")
    event_table = get_event_table(config.events_df_path)
    print(event_table.head())

    print("Getting old and new panddas...")
    original_panddas = get_old_panddas(event_table)
    new_panddas = get_new_panddas(config.new_panddas_dir)
    print("Got {} original panddas".format(len(original_panddas)))
    print("Got {} new panddas".format(len(new_panddas)))

    print("Getting pandda matches")
    pandda_matches = match_panddas(original_panddas,
                                   new_panddas,
                                   )
    print("\tFound {} matches!".format(pandda_matches))

    print("Getting event mappings...")
    if not output.comparison_json_path.exists():
        comparisons = {}
        for original_pandda, new_pandda in pandda_matches:
            print("\tGetting event mapping for: {} and {}".format(original_pandda.pandda_dir.name,
                                                                  new_pandda.pandda_dir.name,
                                                                  )
                  )
            event_mapping = EventMapping(original_pandda, new_pandda)

            comparison = ComparisonSet(original_pandda,
                                       new_pandda,
                                       event_mapping,
                                       )

            comparisons[(original_pandda.pandda_dir, new_pandda.pandda_dir)] = comparison

        print("Saving mappings...")
        # to_json(comparisons,
        #         output_path=output.comparison_json_path,
        #         )
        with open(str(output.comparison_json_path), "wb") as f:
            cloudpickle.dump(comparisons,
                             f,
                             )

    else:
        # comparisons = from_json(pandda_matches,
        #                         output.comparison_json_path,
        #                         )
        with open(str(output.comparison_json_path), "rb") as f:
            comparisons = cloudpickle.load(f)
            print("\tGot {} comparisons".format(len(comparisons)))

    print("Getting comparison dataframe...")
    comparison_df = make_comparison_table(comparisons)
    print(comparison_df.head())

    print("Outputting comparison dataframe to {}".format(output.pandda_comparison_table_path))
    output_comparison_table(comparison_df,
                            output.pandda_comparison_table_path,
                            )


if __name__ == "__main__":
    main()
