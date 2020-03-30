from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

sns.set()


def scatter_plot(x,
                 y,
                 output_path: Path,
                 ):
    ax = sns.scatterplot(x,
                         y,
                         )
    fig = ax.get_figure()
    fig.savefig(str(output_path))
    plt.close(fig)


# def log_scatter_plot(x,
#                  y,
#                  output_path: Path,
#                  ):
#     ax = sns.scatterplot(x,
#                          y,
#                          )
#     fig = ax.get_figure()
#     fig.savefig(str(output_path))
#     plt.close(fig)


def distribution_plot(data_series,
                      output_path: Path,
                      ):
    ax = sns.distplot(data_series)
    fig = ax.get_figure()
    fig.savefig(str(output_path))
    plt.close(fig)


def bar_plot(x, y, output_path):
    ax = sns.barplot(x, y)
    fig = ax.get_figure()
    fig.savefig(str(output_path))
    plt.close(fig)


def cumulative_plot(x,
                    output_path: Path,
                    n_bins=100,
                    ):
    # ax = sns.kdeplot(x,
    #                  cumulative=True,
    #                  )
    # fig = ax.get_figure()
    fig, ax = plt.subplots(figsize=(8, 4))

    n, bins, patches = ax.hist(x,
                               n_bins,
                               density=True,
                               histtype='step',
                               cumulative=True,
                               label='Empirical',
                               )
    fig.savefig(str(output_path))
    plt.close(fig)


def comparitive_cdf_plot(distribution_dict,
                         output_path,
                         n_bins=100,
                         title="",
                         x_label="",
                         y_label="",
                         ):
    fig, ax = plt.subplots(figsize=(8, 4))
    # plt.xscale("log")

    for build_name, distribution in distribution_dict.items():
        n, bins, patches = ax.hist(distribution,
                                   n_bins,
                                   density=True,
                                   histtype='step',
                                   cumulative=True,
                                   label=build_name,
                                   )

    plt.xticks([0, 1, 2, 3, 4, 5],
               [np.exp(0) - 1, np.exp(1) - 1, np.exp(2) - 1, np.exp(3) - 1, np.exp(4) - 1, np.exp(5) - 1],
               )

    ax.legend(loc='right')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.savefig(str(output_path))
    plt.close(fig)
