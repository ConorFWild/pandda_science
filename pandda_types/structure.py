from typing import NamedTuple

import gemmi

class Structure:
    def __init__(self, path):
        self.structure = gemmi.read_structure(str(path))


class Alignment:
    pass


class Residue:
    pass