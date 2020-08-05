import re

import numpy as np
import pandas as pd

import gemmi

from fragalysis_api.xcextracter.getdata import GetPdbData
from fragalysis_api.xcextracter.xcextracter import xcextracter

from pandda_autobuilding.constants import *
from pandda_types import logs


class PanDDAModelPaths:
    def __init__(self, paths):
        self.paths = paths

    def from_dir(self, pandda_dir):
        processed_models_dir = pandda_dir / PANDDA_PROCESSED_DATASETS_DIR
        processed_model_dirs = processed_models_dir.glob("*")

        paths = {}
        for processed_model_dir in processed_model_dirs:
            dtag = processed_model_dir.name
            model_dir = processed_model_dir / PANDDA_MODELLED_STRUCTURES_DIR
            model_path = model_dir / PANDDA_EVENT_MODEL.format(dtag)
            paths[dtag] = model_path

        return PanDDAModelPaths(paths)


class Structure:
    def __init__(self, structure):
        self.structure = structure

    @staticmethod
    def from_model_path(path):
        structure = gemmi.read_structure(str(path))

        return Structure(structure)


class AutobuiltStructures:
    def __init__(self, structures):
        self.structures = structures

    @staticmethod
    def from_dir(pandda_dir):
        model_paths = PanDDAModelPaths(pandda_dir)

        structures = {}
        for dtag, path in model_paths.paths.items():
            model = Structure.from_model_path(model_paths.paths)

            structures[dtag] = model

        return AutobuiltStructures(structures)


class System:
    def __init__(self, system):
        self.system = system

    @staticmethod
    def from_dtag(dtag):
        regex = "([^-]+)-[^-]+"
        matches = re.findall(regex,
                             dtag,
                             )

        return matches[0]


class ProjectCode:
    def __init__(self, project_code):
        self.project_code = project_code

    @staticmethod
    def from_dir(pandda_dir):
        processed_models_dir = pandda_dir / PANDDA_PROCESSED_DATASETS_DIR

        processed_model_dir = next(processed_models_dir.glob("*"))

        example_dtag = processed_model_dir.name
        print("Example dtag is: {}".format(example_dtag))

        project_name = System(example_dtag).system

        return ProjectCode(project_name)


class Dtag:
    def __init__(self, dtag):
        self.dtag = dtag

    @staticmethod
    def from_protein_code(protein_code):
        regex = "^([0-9a-zA-Z]+[-]+[0-9a-zA-Z]+)"
        matches = re.findall(regex,
                             protein_code,
                             )

        return Dtag(matches[0])


class ReferenceStructures:
    def __init__(self):
        pass

    @staticmethod
    def from_dir(pandda_dir):
        project_code = ProjectCode.from_dir(pandda_dir)

        xcd = xcextracter(project_code)

        pdb_grabber = GetPdbData()

        structures = {}
        for index, row in xcd.iterrows():
            protein_code = row["protein_code"]
            dtag = Dtag.from_protein_code(protein_code)
            pdb_block = pdb_grabber.get_bound_pdb_file(protein_code)

            try:
                Structure(pdb_block)

            except:
                None

        return ReferenceStructures(structures)


class Distance:
    def __init__(self):
        pass

    def from_ligands(self):
        pass


class Ligand:
    def __init__(self):
        pass

    @staticmethod
    def from_structure(structure):
        pass


class StructureIterator:
    def __init__(self):
        pass

    def from_structures(self, autobuilt_structures, reference_structures):
        return StructureIterator()


class Distances:
    def __init__(self, distances):
        self.distances = distances

    @staticmethod
    def from_structures(reference_structures, autobuilt_structures):
        distances = {}

        for dtag, autobuilt_structure, reference_structure in StructureIterator(autobuilt_structures,
                                                                                reference_structures):
            autobuild_ligand = Ligand.from_structure(autobuilt_structure)
            reference_ligand = Ligand.from_structure(reference_structure)

            distance = Distance.from_ligands(autobuild_ligand, reference_ligand)

        return Distances(distances)


class AutobuildRMSDTable:
    def __init__(self):
        pass

    @staticmethod
    def from_directory(pandda_dir):
        autobuilt_structures = AutobuiltStructures.from_dir(pandda_dir)

        reference_structures = ReferenceStructures.from_dir(pandda_dir)

        distances = Distances.from_structures(reference_structures, autobuilt_structures)

        table = pd.DataFrame(distances.distances)

        return AutobuildRMSDTable(table)
