from __future__ import annotations

from typing import *

from dataclasses import dataclass

import os
import argparse
import re
import subprocess
import shutil
from pathlib import Path
from pprint import PrettyPrinter

import numpy as np
import pandas as pd

import joblib

from autobuilding_paper.autobuild_constants import *

import gemmi

printer = PrettyPrinter(indent=1)


class Option:
    def __init__(self, short, long, required=False, default=None, help=""):
        self.short = short
        self.long = long
        self.required = required
        self.default = default
        self.help = help

    def __call__(self, args):
        return vars(args)[self.long]


class Config:
    def __init__(self, args=None):
        input_pandda_dir = Option("i", "input_pandda_dir", required=True)
        overwrite = Option("o", "overwrite", required=True)
        pandda_version = Option("p", "pandda_version", required=True)

        options = [input_pandda_dir, overwrite, pandda_version]

        args = self.get_args(options, args)

        self.input_pandda_dir = Path(input_pandda_dir(args))
        self.overwrite = bool(args.overwrite)
        self.pandda_version = int(args.pandda_version)

    def get_args(self, options, args):
        parser = argparse.ArgumentParser()
        for option in options:
            parser.add_argument("-{}".format(option.short),
                                "--{}".format(option.long),
                                help=option.help,
                                required=option.required,
                                default=option.default,
                                )
        if args:
            return parser.parse_args(args)
        else:
            return parser.parse_args()


class PanDDAFilesystemModel:
    def __init__(self, pandda_root_dir: Path):
        self.pandda_root_dir = pandda_root_dir

        self.pandda_analyse_dir = pandda_root_dir / "analyses"
        self.pandda_analyse_events_path = self.pandda_analyse_dir / "pandda_analyse_events.csv"
        self.autobuilding_results_table = self.pandda_analyse_dir / "autobuilding_results.csv"

        self.pandda_processed_datasets_dir = pandda_root_dir / "processed_datasets"
        self.pandda_processed_datasets_dirs = list(self.pandda_processed_datasets_dir.glob("*"))


@dataclass()
class Event:
    event_id: EventID
    centroid: Tuple[float, float, float]
    event_dir: Path
    event_mtz_file: Path
    event_ccp4_file: Path
    initial_pdb_file: Path


@dataclass()
class Events:
    events: Dict[EventID, Event]

    def __iter__(self):
        for event_id in self.events:
            yield event_id

    def __getitem__(self, item):
        return self.events[item]

    @staticmethod
    def from_fs(fs: PanDDAFilesystemModel, pandda_version: int) -> Events:
        # Get event table
        pandda_analyse_events_file = fs.pandda_analyse_events_path
        event_table = pd.read_csv(str(pandda_analyse_events_file))

        # Get events
        events: Dict[EventID, Event] = {}
        for index, row in event_table.iterrows():
            dtag = Dtag(row["dtag"])
            event_idx = EventIDX(row["event_idx"])
            event_id = EventID(dtag, event_idx)

            occupancy = row["1-BDC"]

            centroid = (float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                        )

            event_dir = fs.pandda_processed_datasets_dir / f"{dtag.dtag}"

            event_mtz_file = event_dir / PANDDA_MTZ_FILE.format(dtag.dtag)

            event_ccp4_file = event_dir / PANDDA_EVENT_MAP_FILE.format(dtag=dtag.dtag,
                                                                       event_idx=event_idx.event_idx,
                                                                       bdc=occupancy,
                                                                       )

            initial_pdb_file = event_dir / PANDDA_PDB_FILE.format(dtag.dtag)

            event: Event = Event(event_id=event_id,
                                 centroid=centroid,
                                 event_dir=event_dir,
                                 event_mtz_file=event_mtz_file,
                                 event_ccp4_file=event_ccp4_file,
                                 initial_pdb_file=initial_pdb_file,
                                 )
            events[event_id] = event

        return Events(events)

    def __len__(self):
        return len(self.events)


@dataclass()
class EventDir:
    event_dir: Path

    @classmethod
    def from_event(cls, event: Event):
        return EventDir(event.event_dir)


@dataclass()
class MtzFile:
    mtz_file: Path

    @classmethod
    def from_event(cls, event: Event):
        return MtzFile(event.event_mtz_file)

    @classmethod
    def from_reflections(cls, event_mtz):
        pass

    def save(self, event_mtz: Reflections):
        event_mtz.reflections.write_to_file(str(self.mtz_file))


@dataclass()
class PdbFile:
    pdb_file: Path

    @classmethod
    def from_event(cls, event: Event):
        return PdbFile(event.initial_pdb_file)

    def save(self, structure: Structure):
        structure.structure.write_minimal_pdb(str(self.pdb_file))



@dataclass()
class Ccp4File:
    ccp4_file: Path

    @classmethod
    def from_event(cls, event: Event):
        return Ccp4File(event.event_ccp4_file)


@dataclass()
class SmilesFile:
    smiles_file: Path

    @classmethod
    def from_event(cls, event: Event):
        compound_dir = event.event_dir / PANDDA_LIGAND_FILES_DIR

        # ligand_pdbs = list(compound_dir.glob("*.pdb"))
        # ligand_pdb_strings = [str(ligand_path) for ligand_path in ligand_pdbs if ligand_path.name != "tmp.pdb"]
        # if len(ligand_pdb_strings) > 0:
        #     shortest_ligand_path = min(ligand_pdb_strings,
        #                                key=len,
        #                                )
        #     path = Path(shortest_ligand_path)
        #     return SmilesFile(path)

        # Check for smiles files
        smiles_paths = compound_dir.glob("*.smiles")
        smiles_paths_list = list(smiles_paths)

        if len(smiles_paths_list) > 0:
            path = Path(min([str(ligand_path) for ligand_path in smiles_paths_list if ligand_path.name != "tmp.smiles"],
                            key=len)
                        )
            return SmilesFile(path)

        else:
            exception = f"""
            Looked in {compound_dir} for smiles
            No smiles found! Smiles list is: {smiles_paths_list}
            """
            raise Exception(exception)


@dataclass()
class CifFile:
    cif_file: Path

    @classmethod
    def from_smiles_file(cls, smiles_file: SmilesFile,
                         event_dir: EventDir,
                         autobuilding_ligand: str = "autobuilding_ligand") -> CifFile:
        command = f"module load phenix; cd {event_dir.event_dir}; phenix.elbow {smiles_file.smiles_file} --output=\"{autobuilding_ligand}\""

        p = subprocess.Popen(command,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             )

        stdout, stderr = p.communicate()

        return CifFile(event_dir.event_dir / f"{autobuilding_ligand}.cif")


@dataclass()
class Structure:
    structure: gemmi.Structure

    @classmethod
    def from_pdb_file(cls, pdb_file: PdbFile):
        structure = gemmi.read_structure(str(pdb_file.pdb_file))
        return Structure(structure)

    def strip(self, event: Event, radius: float = 7.0):
        event_centoid = gemmi.Position(*event.centroid)

        new_structure = gemmi.Structure()

        for model_i, model in enumerate(self.structure):
            new_model = gemmi.Model(model.name)
            new_structure.add_model(new_model, pos=-1)

            for chain_i, chain in enumerate(model):
                new_chain = gemmi.Chain(chain.name)
                new_structure[model_i].add_chain(new_chain, pos=-1)

                for residue_i, residue in enumerate(chain):
                    new_residue = gemmi.Residue()
                    new_residue.name = residue.name
                    new_residue.seqid = residue.seqid
                    new_residue.subchain = residue.subchain
                    new_residue.label_seq = residue.label_seq
                    new_residue.het_flag = residue.het_flag
                    new_structure[model_i][chain_i].add_residue(new_residue, pos=-1)

                    for atom_i, atom in enumerate(residue):
                        pos = atom.pos
                        if pos.dist(event_centoid) > radius:
                            new_structure[model_i][chain_i][residue_i].add_atom(atom, pos=-1)

        for model_i, model in enumerate(self.structure):
            self.structure.add_model(new_structure[model_i], pos=-1)
            del self.structure[0]



        return Structure(self.structure)


@dataclass()
class Xmap:
    xmap: gemmi.FloatGrid

    @classmethod
    def from_file(cls, event_map_file: Ccp4File) -> Xmap:
        m = gemmi.read_ccp4_map(str(event_map_file.ccp4_file))
        m.setup()

        grid_array = np.array(m.grid, copy=True)

        new_grid = gemmi.FloatGrid(*grid_array.shape)
        new_grid.spacegroup = m.grid.spacegroup  # gemmi.find_spacegroup_by_name("P 1")
        new_grid.set_unit_cell(m.grid.unit_cell)

        new_grid_array = np.array(new_grid, copy=False)
        new_grid_array[:, :, :] = grid_array[:, :, :]

        return Xmap(new_grid)

    def mask_event(self, event: Event, radius=3.0):
        event_centroid = gemmi.Position(*event.centroid)

        xmap_array = np.array(self.xmap, copy=True)

        mask_grid = gemmi.Int8Grid(*xmap_array.shape)
        mask_grid.spacegroup = self.xmap.spacegroup
        mask_grid.set_unit_cell(self.xmap.unit_cell)

        mask_grid.set_points_around(event_centroid,
                                    radius=radius,
                                    value=1,
                                    )
        mask_grid.symmetrize_max()

        mask_array = np.array(mask_grid, copy=False, dtype=np.int8)

        new_grid = gemmi.FloatGrid(*xmap_array.shape)
        new_grid.spacegroup = self.xmap.spacegroup  # gemmi.find_spacegroup_by_name("P 1")
        new_grid.set_unit_cell(self.xmap.unit_cell)

        new_grid_array = np.array(new_grid, copy=False)

        new_grid_array[np.nonzero(mask_array)] = xmap_array[np.nonzero(mask_array)]
        new_grid.symmetrize_max()

        return Xmap(new_grid)


def array_to_index(event_mtz):
    event_data = np.array(event_mtz, copy=False)
    array_to_index_map = {}
    for i in range(event_data.shape[0]):
        h = int(event_data[i, 0])
        k = int(event_data[i, 1])
        l = int(event_data[i, 2])
        array_to_index_map[i] = (h, k, l)

    return array_to_index_map


def index_to_array(intial_mtz):
    event_data = np.array(intial_mtz, copy=False)
    index_to_array_map = {}
    for i in range(event_data.shape[0]):
        h = int(event_data[i, 0])
        k = int(event_data[i, 1])
        l = int(event_data[i, 2])
        index_to_array_map[(h, k, l)] = i

    return index_to_array_map


@dataclass()
class HKL:
    h: int
    k: int
    l: int

    @staticmethod
    def from_hkl(h, k, l):
        return HKL(h, k, l)

    def __hash__(self):
        return hash((self.h, self.k, self.l))

    def to_list(self):
        return [self.h, self.k, self.l]

    @staticmethod
    def from_list(hkl_list):
        return HKL(hkl_list[0],
                   hkl_list[1],
                   hkl_list[2],
                   )

    def is_000(self):
        if self.h == 0:
            if self.k == 0:
                if self.l == 0:
                    return True

        return False


@dataclass()
class Reflection:
    hkl: HKL
    data: np.array

    @staticmethod
    def from_row(row: np.array):
        h = row[0]
        k = row[1]
        l = row[2]
        hkl = HKL(h, k, l)
        data = row[3:]
        return Reflection(hkl,
                          data)


@dataclass()
class ReflectionsDict:
    reflections_dict: Dict[HKL, Reflection]

    @staticmethod
    def from_array(array):
        reflections = {}
        for row in array:
            reflection = Reflection.from_row(row)
            reflections[reflection.hkl] = reflection

        return ReflectionsDict(reflections)

    def __getitem__(self, item):
        return self.reflections_dict[item]

    def __iter__(self):
        for hkl in self.reflections_dict:
            yield hkl

    def to_array(self):
        rows = []
        for hkl in self.reflections_dict:
            hkl_array = np.array(hkl.to_list())
            data = self.reflections_dict[hkl].data

            row = np.hstack([hkl_array, data])

            rows.append(row)

        array = np.vstack(rows)

        return array


@dataclass()
class Reflections:
    reflections: gemmi.mtz

    @classmethod
    def from_file(cls, initial_mtz_file: MtzFile):
        reflections = gemmi.read_mtz_file(str(initial_mtz_file.mtz_file))
        return Reflections(reflections)

    @classmethod
    def from_xmap(cls, masked_event_map: Xmap, inital_mtz: Reflections):
        masked_event_map.xmap.spacegroup = inital_mtz.reflections.spacegroup
        masked_event_map.xmap.symmetrize_max()

        sf = gemmi.transform_map_to_f_phi(masked_event_map.xmap, half_l=False)
        data = sf.prepare_asu_data(dmin=inital_mtz.resolution(), with_000=True)

        mtz = gemmi.Mtz(with_base=True)
        mtz.spacegroup = sf.spacegroup
        mtz.cell = sf.unit_cell
        mtz.add_dataset('unknown')
        mtz.add_column('FWT', 'F')
        mtz.add_column('PHWT', 'P')
        mtz.set_data(data)

        return Reflections(mtz)

    def merge_mtzs(self, event_map_mtz: Reflections) -> Reflections:
        initial_mtz = self.reflections
        event_mtz = event_map_mtz.reflections

        initial_mtz_data = np.array(initial_mtz, copy=False)
        event_mtz_data = np.array(event_mtz, copy=False)

        initial_reflections: ReflectionsDict = ReflectionsDict.from_array(initial_mtz_data)
        event_reflections: ReflectionsDict = ReflectionsDict.from_array(event_mtz_data)

        # printer.pprint("Initial")
        # printer.pprint(initial_reflections.reflections_dict)
        # printer.pprint("Event")
        # printer.pprint(event_reflections.reflections_dict)

        initial_asu = gemmi.ReciprocalAsu(initial_mtz.spacegroup)
        operations = initial_mtz.spacegroup.operations()

        initial_mtz_fwt_index = initial_mtz.column_labels().index("FWT")
        event_mtz_fwt_index = event_mtz.column_labels().index("FWT")

        initial_mtz_phwt_index = initial_mtz.column_labels().index("PHWT")
        event_mtz_phwt_index = event_mtz.column_labels().index("PHWT")

        fom_index = initial_mtz.column_labels().index("FOM")
        initial_mtz_fo_index = initial_mtz.column_labels().index("F")
        initial_mtz_fc_index = initial_mtz.column_labels().index("FC")
        initial_mtz_phc_index = initial_mtz.column_labels().index("PHIC")
        initial_mtz_r_index = initial_mtz.column_labels().index("FreeR_flag")
        initial_mtz_ls_fc_all_index = initial_mtz.column_labels().index("FC_ALL_LS")
        initial_mtz_ls_phc_all_index = initial_mtz.column_labels().index("PHIC_ALL_LS")
        initial_mtz_fc_all_index = initial_mtz.column_labels().index("FC_ALL")
        initial_mtz_phc_all_index = initial_mtz.column_labels().index("PHIC_ALL")
        initial_mtz_delfwt_index = initial_mtz.column_labels().index("DELFWT")
        initial_mtz_phdelwt_index = initial_mtz.column_labels().index("PHDELWT")

        initial_mtz_sigf_index = initial_mtz.column_labels().index("SIGF")

        print("\tBeginning graft...")
        new_reflections = {}
        for hkl in event_reflections:
            event_reflection = event_reflections[hkl]

            asu_hkl = HKL.from_list(initial_asu.to_asu(hkl.to_list(), operations, ))
            if asu_hkl.is_000():
                data = np.zeros(len(list(initial_reflections.reflections_dict.values())[0].data))
                new_reflection = Reflection(hkl, data)

            elif asu_hkl not in initial_reflections:
                print(f"\tMissing reflection: {asu_hkl}")
                continue

            else:
                initial_reflection: Reflection = initial_reflections[asu_hkl]

                new_reflection = Reflection(hkl, np.copy(initial_reflection.data))

            new_reflection.data[initial_mtz_fwt_index - 3] = event_reflection.data[event_mtz_fwt_index - 3]
            new_reflection.data[initial_mtz_phwt_index - 3] = event_reflection.data[event_mtz_phwt_index - 3]

            new_reflections[hkl] = new_reflection

        new_array = ReflectionsDict(new_reflections).to_array()

        initial_mtz.spacegroup = event_mtz.spacegroup
        initial_mtz.set_data(new_array)

        return Reflections(initial_mtz)

    def resolution(self):
        return self.reflections.resolution_high()


@dataclass()
class RhofitDir:
    rhofit_dir: Path

    @classmethod
    def from_rhofit(cls, event_mtz_file: MtzFile, ligand_cif_file: CifFile, stripped_pdb_file: PdbFile, event: Event):
        env = AUTOBUILD_ENV
        ligand_fit_command = AUTOBUILD_COMMAND
        rhofit_dir = event.event_dir / RHOFIT_EVENT_DIR.format(event.event_id.event_idx.event_idx)
        ligand_fit_args = AUTOBUILD_ARGS.format(mtz=event_mtz_file.mtz_file,
                                                pdb=stripped_pdb_file.pdb_file,
                                                ligand=ligand_cif_file.cif_file,
                                                out_dir_path=rhofit_dir,
                                                )

        command = "{env}; {ligand_fit_command} {args}".format(env=env,
                                                              ligand_fit_command=ligand_fit_command,
                                                              args=ligand_fit_args,
                                                              )

        print(command)

        submit_proc = subprocess.Popen(str(command),
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       )
        stdout, stderr = submit_proc.communicate()

        return RhofitDir(rhofit_dir)


@dataclass()
class BuildNumberID:
    build_number_id: int

    def __hash__(self):
        return hash(self.build_number_id)


@dataclass()
class BuildClusterID:
    build_cluster_id: int

    def __hash__(self):
        return hash(self.build_cluster_id)


@dataclass()
class Build:
    build_file: Path
    build_rscc: float


@dataclass()
class EventIDX:
    event_idx: int

    def __hash__(self):
        return hash(self.event_idx)


@dataclass()
class Dtag:
    dtag: str

    def __hash__(self):
        return hash(self.dtag)


@dataclass()
class EventID:
    dtag: Dtag
    event_idx: EventIDX

    def __hash__(self):
        return hash((self.dtag, self.event_idx))


@dataclass()
class ClusterBuildResults:
    cluster_build_results: Dict[BuildNumberID, Build]

    def __iter__(self):
        for build_number_id in self.cluster_build_results:
            yield build_number_id

    def __getitem__(self, item):
        return self.cluster_build_results[item]


@dataclass()
class EventBuildResults:
    build_results: Dict[BuildClusterID, ClusterBuildResults]

    def __iter__(self):
        for build_cluster_id in self.build_results:
            yield build_cluster_id

    def __getitem__(self, item):
        return self.build_results[item]

    @classmethod
    def from_rhofit_dir(cls, rhofit_dir: RhofitDir, event: Event):

        with open(str(rhofit_dir.rhofit_dir / RHOFIT_RESULTS_FILE), "r") as f:
            results_string = f.read()

        build_matches = re.findall(RHOFIT_HIT_REGEX,
                                   results_string,
                                   )

        cluster_builds = {}
        for build_match in build_matches:
            build_file = build_match[0]
            rscc = build_match[1]

            cluster_build_matches = re.findall(RHOFIT_CLUSTER_BUILD_REGEX,
                                               build_file,
                                               )
            cluster = BuildClusterID(cluster_build_matches[0][0])
            build_number = BuildNumberID(cluster_build_matches[0][1])

            if cluster not in cluster_builds:
                cluster_builds[cluster] = {}

            cluster_builds[cluster][build_number] = Build(build_file=build_file,
                                                          build_rscc=rscc,
                                                          )

        return EventBuildResults.from_dict(cluster_builds)

    @classmethod
    def from_dict(cls, cluster_builds_dict: Dict[BuildClusterID, Dict[BuildNumberID, Build]]):
        event_clusters = {}
        for cluster_id in cluster_builds_dict:
            cluster_builds = cluster_builds_dict[cluster_id]
            event_clusters[cluster_id] = ClusterBuildResults(cluster_builds)

        return EventBuildResults(event_clusters)


@dataclass()
class DtagBuildResults:
    build_results: Dict[EventIDX, EventBuildResults]

    def __iter__(self):
        for event_idx in self.build_results:
            yield event_idx

    def __getitem__(self, item):
        return self.build_results[item]


@dataclass()
class AutobuildingResults:
    autobuilding_results: Dict[Dtag, DtagBuildResults]

    @classmethod
    def from_rhofit_dir(cls, rhofit_dir, event):
        pass

    def __iter__(self):
        for dtag in self.autobuilding_results:
            yield dtag

    def __getitem__(self, item):
        return self.autobuilding_results[item]

    def get_best_event_builds(self):
        best_event_builds = {}
        for dtag in self:
            dtag_builds: DtagBuildResults = self[dtag]
            for event_id in dtag_builds:
                event_builds: EventBuildResults = dtag_builds[event_id]

                event_builds_flat = {}
                for build_cluster_id in event_builds:
                    cluster_builds: ClusterBuildResults = event_builds[build_cluster_id]

                    for build_id in cluster_builds:
                        build: Build = cluster_builds[build_id]
                        event_builds_flat[(build_cluster_id, build_id)] = build

                best_event_build: Tuple[BuildClusterID, BuildNumberID] = min(event_builds_flat,
                                                                             key=lambda build_id: event_builds_flat[
                                                                                 build_id].build_rscc,
                                                                             )

                new_cluster_builds: ClusterBuildResults = ClusterBuildResults(
                    {best_event_build[1]: event_builds_flat[best_event_build]})
                new_event_buils: EventBuildResults({best_event_build[0]: new_cluster_builds})

                best_event_builds[(dtag, event_id)] = best_event_build

    @classmethod
    def from_event_build_results(cls, event_autobuilding_results):
        pass

    def to_flat_dict(self):
        builds = {}

        for dtag in self:
            if dtag not in builds:
                builds[dtag] = {}

            event_builds = self[dtag]
            for event_id in event_builds:
                if event_id not in builds[dtag]:
                    builds[dtag][event_id] = {}

                cluster_builds = event_builds[event_id]
                for cluster_id in event_builds:
                    if cluster_id not in cluster_builds[cluster_id]:
                        builds[dtag][event_id][cluster_id] = {}

                    for build_number in cluster_builds[cluster_id]:
                        builds[(dtag, event_id, cluster_id, build_number)] = cluster_builds[build_number]

        return builds
