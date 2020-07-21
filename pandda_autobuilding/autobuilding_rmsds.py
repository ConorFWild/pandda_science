import os
import argparse
from pathlib import Path

import pandas as pd

from pandda_autobuilding.fragalysis import (get_autobuild_rmsds)


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

def main():
    config = Config()

    table = get_autobuild_rmsds(config.input_pandda)
    print(table)




if __name__ == "__main__":
    main()