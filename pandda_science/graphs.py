from pathlib import Path

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
