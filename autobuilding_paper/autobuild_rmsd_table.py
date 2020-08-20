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
        print("Got {} autobuilt structures".format(autobuilt_structures))

        reference_structures = ReferenceStructures.from_dir(pandda_dir)

        distances = RMSDs.from_structures(reference_structures, autobuilt_structures)

        table = pd.DataFrame(distances.distances)

        return AutobuildRMSDTable(table)
