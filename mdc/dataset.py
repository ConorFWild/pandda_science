import gemmi

from mdc.reflections import PanDDAReflections
from mdc.structure import PanDDAStructure


class PanDDADataset:
    def __init__(self, id, reflections, structure):
        self.id = id
        self.reflections = reflections
        self.structure = structure

    @staticmethod
    def from_dir(path, mtz_regex="dimple.mtz", pdb_regex="dimple.pdb"):
        id = path.name
        reflection_path = path / mtz_regex
        structure_path = path / pdb_regex
        return PanDDADataset.from_paths(id, reflection_path, structure_path)

    @staticmethod
    def from_paths(id, reflections_path, structure_path):
        reflections = PanDDAReflections.from_file(reflections_path)
        structure = PanDDAStructure.from_file(structure_path)

        return PanDDADataset(id, reflections, structure)

    def get_resolution_high(self):
        return self.reflections.get_resolution()
