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
        regex = "([^\-]+)\-[^\-]+"
        # regex = "([0-9])"
        print("\tdtag to match: {}".format(dtag))
        matches = re.findall(regex,
                             str(dtag),
                             )
        print([match for match in matches])

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

        project_name = System.from_dtag(example_dtag).system

        return ProjectCode(project_name)


@dataclasses.dataclass()
class Dtag:
    dtag: str

    @staticmethod
    def from_protein_code(protein_code):
        regex = "^([0-9a-zA-Z]+[-]+[0-9a-zA-Z]+)"
        matches = re.findall(regex,
                             protein_code,
                             )

        return Dtag(matches[0])


@dataclasses.dataclass()
class EventIDX:
    event_idx: int


class ReferenceStructures:
    def __init__(self, structures):
        self.structures = structures

    @staticmethod
    def from_dir(pandda_dir):
        project_code = ProjectCode.from_dir(pandda_dir)
        print("\tProject code is: {}".format(project_code.project_code))

        xcd = xcextracter(project_code.project_code)

        pdb_grabber = GetPdbData()

        structures = {}
        for index, row in xcd.iterrows():
            protein_code = row["protein_code"]
            dtag = Dtag.from_protein_code(protein_code)
            pdb_block = pdb_grabber.get_bound_pdb_file(protein_code)

            try:
                Structure(pdb_block)

            except Exception as e:
                continue

        return ReferenceStructures(structures)


class Distance:
    def __init__(self):
        pass

    def from_ligands(self):
        pass


@dataclasses.dataclass()
class LigandResidues:
    residues: typing.List[typing.Any]

    @staticmethod
    def from_structure(structure):
        residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.name == "LIG":
                        residues.append(residue)

        return LigandResidues(residues)


class StructureIterator:
    def __init__(self):
        pass

    def from_structures(self, autobuilt_structures, reference_structures):
        return StructureIterator()


# class Distances:
#     def __init__(self, distances):
#         self.distances = distances
#
#     @staticmethod
#     def from_structures(reference_structures, autobuilt_structures):
#         distances = {}
#
#         for dtag, autobuilt_structure, reference_structure in StructureIterator(autobuilt_structures,
#                                                                                 reference_structures,
#                                                                                 ):
#             autobuild_ligand = LigandResidues.from_structure(autobuilt_structure)
#             reference_ligand = LigandResidues.from_structure(reference_structure)
#
#
#
#         return Distances(distances)


@dataclasses.dataclass()
class RMSD:
    rmsd: float

    @staticmethod
    def from_residues(residue_1, residue_2):
        closest_distances = []
        for atom_1 in residue_1:
            distances = []
            for atom_2 in residue_2:
                distance = atom_1.pos.dist(atom_2.pos)
                distances.append(distance)

            closest_distance = min(distances)
            closest_distances.append(closest_distance)

        mean_distance = float(np.mean(closest_distances))
        return RMSD(mean_distance)


@dataclasses.dataclass()
class CommonKeyIterator(typing.Iterable):
    dict_1: typing.Any
    dict_2: typing.Any

    def __next__(self):
        for key in self.dict_1:
            if key in self.dict_2:
                yield key, self.dict_1[key], self.dict_2[key]

    def __iter__(self):
        return self


@dataclasses.dataclass()
class RMSDs:
    distances: typing.Dict[Dtag, RMSD]

    def __getitem__(self, item):
        return self.distances[item]

    def __setitem__(self, key, value):
        self.distances[key] = value

    @staticmethod
    def from_structures(reference_structures, autobuilt_structures):
        rmsds = {}

        for dtag, autobuilt_structure, reference_structure in CommonKeyIterator(autobuilt_structures,
                                                                                reference_structures):
            autobuild_ligand = LigandResidues.from_structure(autobuilt_structure)
            reference_ligand = LigandResidues.from_structure(reference_structure)

            residue_rmsds = []
            for residue_1 in autobuild_ligand.residues:
                for residue_2 in reference_ligand.residues:
                    rmsd = RMSD.from_residues(residue_1,
                                              residue_2,
                                              )

                    residue_rmsds.append(rmsd.rmsd)

            rmsds[dtag] = RMSD(min(residue_rmsds))

        return RMSDs(rmsds)


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
        return self.dtag, self.event_idx


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
