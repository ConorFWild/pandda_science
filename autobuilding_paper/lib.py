import re

import dataclasses
import typing

import numpy as np
import pandas as pd

import gemmi

from fragalysis_api.xcextracter.getdata import GetPdbData
from fragalysis_api.xcextracter.xcextracter import xcextracter

from autobuilding_paper.constants import *


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
    def __init__(self, structures):
        self.structures = structures

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


class ResidueEventDistance:
    def __init__(self, distance):
        self.distance = distance

    @staticmethod
    def from_residue_event(residue,
                           event,
                           ):
        fragment_mean_coord = (ResidueMeanCoord.from_residue(residue)).to_euclidean_coord()
        event_centroid_coord = event.to_centroid_coord()

        distance = (EuclideanDistance.from_coords(fragment_mean_coord,
                                                  event_centroid_coord,
                                                  )
                    ).to_float()

        return distance

    def to_float(self):
        return self.distance


@dataclasses.dataclass()
class EuclideanDistance:
    distance: float

    @staticmethod
    def from_coords(coord_1, coord_2):
        vec_1 = coord_1.to_array()
        vec_2 = coord_2.to_array()

        distance = np.linalg.norm(vec_1 - vec_2)

        return EuclideanDistance(distance)

    def to_float(self):
        return self.distance


@dataclasses.dataclass()
class EuclideanCoord:
    x: float
    y: float
    z: float

    def to_array(self):
        return np.array([self.x, self.y, self.z])


@dataclasses.dataclass()
class ResidueMeanCoord:
    coord: EuclideanCoord

    @staticmethod
    def from_residue(residue):
        coords = []
        for atom in residue:
            pos = atom.pos
            coords.append([pos.x, pos.y, pos.z])

        coords_array = np.array(coords)
        mean_coords = np.mean(coords_array,
                              axis=0,
                              )

        coord = EuclideanCoord(x=mean_coords[0],
                               y=mean_coords[1],
                               z=mean_coords[2],
                               )
        return ResidueMeanCoord(coord)

    def to_euclidean_coord(self):
        return self.coord


@dataclasses.dataclass()
class Ligands:
    ligands: typing.List[gemmi.Residue]

    @staticmethod
    def from_structure(structure):
        ligands = []

        for model in structure:
            for chain in model:
                for res in chain:
                    if res.name == "LIG":
                        ligands.append(res)

        return Ligands(ligands)


@dataclasses.dataclass()
class EventID:
    dtag: str
    event_idx: int

    def __hash__(self):
        return (self.dtag, self.event_idx)


@dataclasses.dataclass()
class Event:
    dtag: str
    event_idx: int
    x: float
    y: float
    z: float

    @staticmethod
    def from_dict(dictionary):
        return Event(dtag=dictionary["dtag"],
                     event_idx=dictionary["event_idx"],
                     x=dictionary["x"],
                     y=dictionary["y"],
                     z=dictionary["z"],
                     )


@dataclasses.dataclass()
class PanDDAEvents:
    dictionary: typing.Dict[EventID, Event]

    def __getitem__(self, item):
        return self.dictionary[item]

    def __setitem__(self, key, value):
        self.dictionary[key] = value

    @staticmethod
    def from_dir(pandda_dir):
        pandda_analyse_file = pandda_dir / PANDDA_ANALYSES_DIR / PANDDA_ANALYSE_EVENTS_FILE

        table = pd.read_csv(str(pandda_analyse_file))

        events: typing.Dict[EventID, Event] = {}
        for index, series in table.iterrows():
            event_id = EventID(dtag=series["dtag"],
                               event_idx=series["event_idx"],
                               )
            events[event_id] = Event.from_dict(series)

        return PanDDAEvents(dictionary=events)
