from pathlib import Path

import seaborn as sns

sns.set()


def scatter_plot(x, y,
            output_path: Path, ):
    ax = sns.scatterplot(x,
                         y,
                         )
    fig = ax.get_figure()
    fig.savefig(str(output_path))
    fig.close()


def distribution_plot(data_series,
                      output_path: Path,
                      ):
    ax = sns.distplot(data_series)
    fig = ax.get_figure()
    fig.savefig(str(output_path))
    fig.close()
