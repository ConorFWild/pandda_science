from typing import NamedTuple, List, Dict
import os
import shutil
import argparse
from pathlib import Path
import json
from functools import partial

import numpy as np
import pandas as pd

import joblib

from pandda_types.data import (Event, )
from pandda_autobuilding.result import Result
from pandda_autobuilding.model import (BioPandasModel, get_rmsd_dfs, )


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-e", "--events_df_path",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-o", "--out_dir",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    parser.add_argument("-a", "--autobuilding_dir_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    args = parser.parse_args()

    return args


class Config(NamedTuple):
    out_dir_path: Path
    events_df_path: Path
    autobuilding_dir_path: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir),
                    events_df_path=Path(args.events_df_path),
                    autobuilding_dir_path=Path(args.autobuilding_dir_path)
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path
        self.results_csv_path = out_dir_path / "autobuilding_results.csv"

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


def get_events(events_csv_path):
    events_table = pd.read_csv(str(events_csv_path))
    events: List[Event] = []

    for idx, event_record in events_table.iterrows():
        event: Event = Event(dtag=str(event_record["dtag"]),
                             event_idx=int(event_record["event_idx"]),
                             occupancy=event_record["occupancy"],
                             analysed_resolution=event_record["analysed_resolution"],
                             high_resolution=event_record["high_resolution"],
                             interesting=event_record["interesting"],
                             ligand_placed=event_record["ligand_placed"],
                             ligand_confidence=event_record["ligand_confidence"],
                             viewed=event_record["viewed"],
                             initial_model_path=Path(event_record["initial_model_path"]),
                             data_path=Path(event_record["data_path"]),
                             final_model_path=Path(event_record["final_model_path"]),
                             event_map_path=Path(event_record["event_map_path"]),
                             actually_built=event_record["actually_built"],
                             x=event_record["x"],
                             y=event_record["y"],
                             z=event_record["z"],
                             )

        events.append(event)

    return events


def is_event_built(event):
    if event.actually_built:
        return True
    else:
        return False


def try_get_json(path):
    try:
        with open(str(path), "r") as f:
            string = f.read()
            results_dict = json.loads(string)
            phenix_control_result = Result(time=float(results_dict["time"]),
                                           success=bool(results_dict["success"]),
                                           result_model_paths=[Path(p)
                                                               for p
                                                               in results_dict[
                                                                   "result_model_paths"]
                                                               ],
                                           )
        return phenix_control_result

    except Exception as e:
        print(e)
        print("\tCouldn't find an results json! Skipping!")
        return None


def analyse_build_result(true_model, result):
    candidate_model_results = []
    distances_to_events = []

    for candidate_model_path in result.result_model_paths:
        candidate_model_record = {}

        result_model: BioPandasModel = BioPandasModel(candidate_model_path)

        # Loop over potential chain

        rmsds = []
        for chain_id in true_model.model.df["HETATM"]["chain_id"].unique():
            # if not is_comparable(true_model, result_model):
            #     print("True model and built model of different lengths! Cannot compare!")
            #     continue
            #

            true_model_df = \
                true_model.model.df["HETATM"][true_model.model.df["HETATM"]["chain_id"] == chain_id][
                    ["x_coord", "y_coord", "z_coord"]]
            result_model_df = \
                result_model.model.df["HETATM"][["x_coord", "y_coord", "z_coord"]]

            true_model_mean_coords = np.mean(np.array(true_model_df), axis=0)
            event_mean_coords = np.array([event.x,
                                          event.y,
                                          event.z,
                                          ])
            distance_from_event_to_model = np.linalg.norm(true_model_mean_coords - event_mean_coords)

            if len(true_model_df) != len(result_model_df):
                continue

            rmsd = get_rmsd_dfs(true_model_df,
                                result_model_df,
                                )

            rmsds.append(rmsd)

            distances_to_events.append(distance_from_event_to_model)

        if len(rmsds) == 0:
            print("\tCOULD NOT COMPARE TO TRUE! SKIPPING!")
            continue

        candidate_model_record["rmsd"] = min(rmsds)

        candidate_model_results.append(candidate_model_record)

    return candidate_model_results


def get_autobuild_record(build_name,
                         candidate_model_results,
                         distance_to_event,
                         result):
    record = {}
    record["dtag"] = event.dtag
    record["event_idx"] = event.event_idx
    record["method"] = build_name
    record["num_candidates"] = len(result.result_model_paths)
    record["min_rmsd"] = min([candidate_model_record["rmsd"]
                              for candidate_model_record
                              in candidate_model_results]
                             )
    record["mean_rmsd"] = np.mean([candidate_model_record["rmsd"]
                                   for candidate_model_record
                                   in candidate_model_results]
                                  )

    record["time"] = result.time
    record["distance_to_event"] = distance_to_event
    return record


def get_null_autobuild_record(build_name):
    record = {}
    record["dtag"] = event.dtag
    record["event_idx"] = event.event_idx
    record["method"] = build_name
    record["num_candidates"] = 0
    record["min_rmsd"] = 0
    record["mean_rmsd"] = 0

    record["time"] = 0
    record["distance_to_event"] = 0
    return record


def analyse_autobuilding_results(true_model_path,
                                 event,
                                 results,
                                 ):
    true_model = BioPandasModel(true_model_path)

    method_results = []

    for build_name, result in results.items():

        candidate_model_results, distance_to_event = analyse_build_result(true_model,
                                                                          result,
                                                                          )

        if len(candidate_model_results) == 0:
            record = get_null_autobuild_record(build_name)

        else:
            record = get_autobuild_record(build_name,
                                          candidate_model_results,
                                          distance_to_event,
                                          result,
                                          )

        method_results.append(record)

    return pd.DataFrame(method_results)


def make_results_dataframe(all_results,
                           true_model_paths,
                           events_dict,
                           ):
    autobuilding_df_tasks = []

    for event_id, true_model_path in true_model_paths.items():
        print("\tAnalysing event for: {}".format(event_id))
        event = events_dict[event_id]
        results = all_results[event_id]

        autobuilding_df_task = partial(analyse_autobuilding_results,
                                        true_model_path,
                                        event,
                                        results,
                                        )
        autobuilding_df_tasks.append(autobuilding_df_task)

    # autobuilding_dfs.append(autobuilding_df)
    print("MULTIPROCESSING")
    autobuilding_dfs = joblib.Parallel(n_jobs=20,
                                       verbose=50)(task
                                                  for task
                                                  in autobuilding_df_tasks)

    df = pd.concat(autobuilding_dfs)

    return df


if __name__ == "__main__":
    args = parse_args()

    config = get_config(args)

    output: Output = setup_output_directory(config.out_dir_path)

    print("Getting events from csv...")
    events = get_events(config.events_df_path)
    print("\tGot events csv with: {} events".format(len(events)))

    results = {}
    final_model_paths = {}

    for event in events:
        print("Processing event: {} {}".format(event.dtag, event.event_idx))

        if not is_event_built(event):
            print("\tNo Model for event at: {}! Skipping!".format(event.final_model_path))
            continue

        event_output_dir_path: Path = config.autobuilding_dir_path / "{}_{}".format(event.dtag,
                                                                                    event.event_idx,
                                                                                    )

        # Get Phenix control

        phenix_control_json_path: Path = event_output_dir_path / "phenix_control.json"
        phenix_control_result = try_get_json(phenix_control_json_path)

        # Get Phenix event
        phenix_event_json_path: Path = event_output_dir_path / "phenix_event.json"
        phenix_event_result = try_get_json(phenix_event_json_path)

        if phenix_event_result or phenix_control_result is None:
            print("\tMissing a json, skipping!")

        results[(event.dtag, event.event_idx)] = {}
        results[(event.dtag, event.event_idx)]["phenix_control"] = phenix_control_result
        results[(event.dtag, event.event_idx)]["phenix_event"] = phenix_event_result
        final_model_paths[(event.dtag, event.event_idx)] = event.final_model_path

    print("\tFinished getting results jsons, with: {} jsons found".format(len(results)))

    print("Making results dataframe")
    events_dict = {(event.dtag, event.event_idx): event
                   for event
                   in events}
    results_df = make_results_dataframe(results,
                                        final_model_paths,
                                        events_dict,
                                        )
    print("\tMade results dataframe")

    print("Outputing results dataframe to: {}".format(output.results_csv_path))
    results_df.to_csv(str(output.results_csv_path))
    print("\tOutput results dataframe")
