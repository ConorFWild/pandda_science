import gemmi

class PanDDAStructure:
    def __init__(self, structure):
        self.structure = structure

    @staticmethod
    def from_file(path):
        structure = gemmi.read_pdb(path)
        return PanDDAStructure(structure)