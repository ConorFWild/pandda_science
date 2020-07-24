import os
import argparse
from pathlib import Path

import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from pandda_autobuilding.fragalysis import (get_autobuild_rmsds)
from pandda_types import logs

RMSD_PLOT_FILE = "rmsds.png"

from pprint import PrettyPrinter


class Config:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-i", "--input_pandda",
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

        self.input_pandda = Path(args.input_pandda)
        self.out_dir_path = Path(args.out_dir_path)


def plot_rmsds(table, path):

    fig, ax = plt.subplots(figsize=(15,15))
    sns.distplot(table["distance"][pd.notna(table["distance"])],
                 rug=True,
                 ax=ax,
                 )

    fig.savefig(str(path))



def main():
    config = Config()

    table = get_autobuild_rmsds(config.input_pandda)
    print(table)
    for index, row in table.iterrows():
        dtag = row["dtag"]
        rmsd = row["distance"]
        logs.LOG[dtag]["distance"] = rmsd

    plot_rmsds(table,
               config.out_dir_path / RMSD_PLOT_FILE,
               )
    print("Saved figure to {}".format(config.out_dir_path / RMSD_PLOT_FILE))

    printer = PrettyPrinter()
    printer.pprint(logs.LOG.dict)

if __name__ == "__main__":
    main()