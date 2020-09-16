import re

import numpy as np
import pandas as pd

from autobuilding_paper.lib import AutobuiltStructures, ReferenceStructures, RMSDs


class AutobuildRMSDTable:
    def __init__(self, table):
        self.table = table

    @staticmethod
    def from_directory(pandda_dir):
        autobuilt_structures = AutobuiltStructures.from_dir(pandda_dir)
        print("Got {} autobuilt structures".format(len(autobuilt_structures.structures)))

        reference_structures = ReferenceStructures.from_dir(pandda_dir)
        print("\t\tGot {} reference structures".format(len(reference_structures.structures)))

        distances = RMSDs.from_structures(reference_structures, autobuilt_structures)
        print("\t\tGot {} distances".format(len(distances)))

        table = pd.DataFrame([{"dtag": dtag.dtag, "rsmd": rmsd.rmsd} for dtag, rmsd in distances.distances.items()])

        return AutobuildRMSDTable(table)
