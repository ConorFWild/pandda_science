from typing import List
from pathlib import Path

import pandas as pd

from pandda_science.graphs import distribution_plot


def get_autobuilding_results_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))

    return parallel_pandda_df


def get_relative_median_rmsd_by_system_df(autobuilding_results_df: pd.DataFrame) -> pd.DataFrame:
    unique_dtags = autobuilding_results_df["dtag"].unique()

    records = []
    for dtag in unique_dtags:
        dataset_df = autobuilding_results_df[autobuilding_results_df["dtag"] == dtag]
        unique_events = dataset_df["event_idx"].unique()

        for event_idx in unique_events:
            dataset_events_df = dataset_df[dataset_df["event_idx"] == event_idx]

            event_distance_to_model = dataset_events_df["distance_to_event"].iloc[0]

            # Phenxi control
            phenix_control_df = dataset_events_df[dataset_events_df["method"] == "phenix_control"]
            phenix_control_rmsd = phenix_control_df["min_rmsd"].iloc[0]
            if phenix_control_rmsd == 0:
                continue

            # Phenix event
            phenix_event_df = dataset_events_df[dataset_events_df["method"] == "phenix_event"]
            phenix_event_rmsd = phenix_event_df["min_rmsd"].iloc[0]
            if phenix_event_rmsd == 0:
                continue

            record = {"dtag": dtag,
                      "event_idx": event_idx,
                      "phenix_control_rmsd": phenix_control_rmsd,
                      "phenix_event_rmsd": phenix_event_rmsd,
                      "event_distance_to_model": event_distance_to_model,
                      }
            records.append(record)
    return pd.DataFrame(records)


def get_autobuilding_rmsd_distribution_graph(autobuilding_rmsd_df: pd.DataFrame,
                                             output_path: Path,
                                             cutoffs: List,
                                             ):
    print("\tWith no cutoff there are: {} autobuilt events...".format(len(autobuilding_rmsd_df)))
    for cutoff in cutoffs:
        cutoff_df = autobuilding_rmsd_df[autobuilding_rmsd_df["event_distance_to_model"] < cutoff]
        print("\t\tAfter cutting off all those events beyond {} A, {} events remain".format(cutoff,
                                                                                            len(cutoff_df),
                                                                                            )
              )
        print(autobuilding_rmsd_df.head())

        # Phenix Control
        distribution_plot(cutoff_df["phenix_control_rmsd"],
                          output_path / "phenix_control_rmsds_{}.png".format(cutoff),
                          )

        # Phenix Event
        distribution_plot(cutoff_df["phenix_event_rmsd"],
                          output_path / "phenix_event_rmsds_{}.png".format(cutoff),
                          )


def get_autobuilding_rscc_distribution_graph():
    raise NotImplementedError()


def get_relative_median_rmsd_by_system_graph():
    raise NotImplementedError()
