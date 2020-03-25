from pathlib import Path

import numpy as np
import pandas as pd

from pandda_science.graphs import distribution_plot, bar_plot


def get_global_cluster_distribution_df(path: Path):
    parallel_pandda_df: pd.DataFrame = pd.read_csv(str(path))

    return parallel_pandda_df


def num_clusters_distribution_histogram(global_cluster_distribution_df,
                                        output_path,
                                        ):
    values, counts = np.unique(global_cluster_distribution_df["num_clusters"],
                               return_counts=True,
                               )

    bar_plot(values,
             counts,
             output_path / "num_clusters.png",
             )


def num_modelable_distribution_histogram(global_cluster_distribution_df,
                                         output_path,
                                         ):
    values, counts = np.unique(global_cluster_distribution_df["num_modelable_clusters"],
                               return_counts=True,
                               )

    distribution_plot(values,
                      counts,
                      output_path / "num_modelable_clusters.png",
                      )
