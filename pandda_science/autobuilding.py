from typing import List
from pathlib import Path

import numpy as np
import pandas as pd

from pandda_science.graphs import (distribution_plot,
                                   scatter_plot,
                                   cumulative_plot,
                                   comparitive_cdf_plot,
                                   )


def get_autobuilding_results_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))
    print("\t{}".format(len(parallel_pandda_df)))
    parallel_pandda_df = parallel_pandda_df.dropna()
    print("\t{}".format(len(parallel_pandda_df)))

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


def get_autobuilding_rmsd_distribution_stats_table(autobuilding_rmsd_df: pd.DataFrame,
                                                   output_path: Path,
                                                   cutoffs: List,
                                                   ):
    records = []
    for cutoff in cutoffs:
        cutoff_df = autobuilding_rmsd_df[autobuilding_rmsd_df["event_distance_to_model"] < cutoff]

        rmsd_difference = cutoff_df["phenix_event_rmsd"] - cutoff_df["phenix_control_rmsd"]
        print("\t\tNumber of builds with lower control rmsd: {}".format(len(rmsd_difference[rmsd_difference > 0])))
        print("\t\tNumber of builds with lower event rmsd: {}".format(len(rmsd_difference[rmsd_difference < 0])))
        print("\t\tMean rmsd difference: {}".format(np.mean(rmsd_difference)))
        print("\t\tMedian rmsd difference: {}".format(np.median(rmsd_difference)))

        print("\t\tMean event rmsd: {}".format(np.mean(cutoff_df["phenix_event_rmsd"])))
        print("\t\tMedian event rmsd: {}".format(np.median(cutoff_df["phenix_event_rmsd"])))

        print("\t\tMean control rmsd: {}".format(np.mean(cutoff_df["phenix_control_rmsd"])))
        print("\t\tMedian control rmsd: {}".format(np.median(cutoff_df["phenix_control_rmsd"])))
        record = {"event_to_model_distance_cutoff": cutoff,
                  "num_control_lower_rmsd_builds": len(rmsd_difference[rmsd_difference > 0]),
                  "num_event_lower_rmsd_builds": len(rmsd_difference[rmsd_difference < 0]),
                  "mean_difference_in_rmsd": np.mean(rmsd_difference),
                  "median_difference_in_rmsd": np.median(rmsd_difference),
                  "phenix_event_mean_rmsd": np.mean(cutoff_df["phenix_event_rmsd"]),
                  "phenix_event_median_rmsd": np.median(cutoff_df["phenix_event_rmsd"]),
                  "phenix_control_mean_rmsd": np.mean(cutoff_df["phenix_control_rmsd"]),
                  "phenix_control_median_rmsd": np.median(cutoff_df["phenix_control_rmsd"])
                  }

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(str(output_path))


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
        print(cutoff_df.head())

        # Phenix Control
        distribution_plot(cutoff_df["phenix_control_rmsd"],
                          output_path / "phenix_control_rmsds_{}.png".format(cutoff),
                          )
        cumulative_plot(cutoff_df["phenix_control_rmsd"],
                        output_path / "phenix_control_rmsds_cumulative_{}.png".format(cutoff),
                        )
        cumulative_plot(np.log(cutoff_df["phenix_control_rmsd"] + 1),
                        output_path / "phenix_control_rmsds_cumulative_log_{}.png".format(cutoff),
                        )

        # Phenix Event
        distribution_plot(cutoff_df["phenix_event_rmsd"],
                          output_path / "phenix_event_rmsds_{}.png".format(cutoff),
                          )
        cumulative_plot(cutoff_df["phenix_event_rmsd"],
                        output_path / "phenix_event_rmsds_cumulative_{}.png".format(cutoff),
                        )
        cumulative_plot(np.log(cutoff_df["phenix_event_rmsd"] + 1),
                        output_path / "phenix_event_rmsds_cumulative_log_{}.png".format(cutoff),
                        )

        # Scatter
        scatter_plot(cutoff_df["phenix_event_rmsd"],
                     cutoff_df["phenix_control_rmsd"],
                     output_path / "phenix_rmsds_scatter_{}.png".format(cutoff),
                     )

        # Log scatter
        scatter_plot(np.log(cutoff_df["phenix_event_rmsd"] + 1),
                     np.log(cutoff_df["phenix_control_rmsd"] + 1),
                     output_path / "phenix_rmsds_log_scatter_{}.png".format(cutoff),
                     )

        # Comparitive log
        comparitive_cdf_plot({"phenix_control": cutoff_df["phenix_control_rmsd"],
                              "phenix_event": cutoff_df["phenix_event_rmsd"],
                              },
                             output_path / "comparitive_rmsd_log_cdf_{}.png".format(cutoff),
                             x_label="RMSD to true model",
                             y_label="Cumulative density",
                             )
        # comparitive_cdf_plot({"phenix_control": cutoff_df["phenix_control_rmsd"] + 1),
        #                       "phenix_event": cutoff_df["phenix_event_rmsd"] ,
        #                       },
        #                      output_path / "comparitive_rmsd_log_cdf_{}.png".format(cutoff),
        #                      )


def get_autobuilding_rscc_distribution_graph():
    raise NotImplementedError()


def get_relative_median_rmsd_by_system_graph():
    raise NotImplementedError()
