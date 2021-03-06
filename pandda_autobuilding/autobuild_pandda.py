from __future__ import annotations
import dataclasses
import typing

import os
import argparse
import re
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import joblib

import gemmi
from biopandas.pdb import PandasPdb


# class Dataset:
#     def __init__(self, dataset_dir, data_path, model_path, compound_path, event_map_paths):
#         self.dataset_dir = dataset_dir
#         self.data_path = data_path
#         self.model_path = model_path
#         self.compound_path = compound_path
#         self.event_map_paths = event_map_paths
#
#     @staticmethod
#     def from_dir(dataset_dir):
#         compound_path = Dataset.get_compound_path(dataset_dir)
#         model_path = Dataset.get_model_path(dataset_dir)
#         data_path = Dataset.get_data_path(dataset_dir)
#         event_map_paths = Dataset.get_event_map_paths(dataset_dir)
#
#         dataset = Dataset(dataset_dir, data_path, model_path, compound_path, event_map_paths)
#         return dataset
class AutobuildingCommandRhofit:
    def __init__(self,
                 out_dir_path=None,
                 mtz_path=None,
                 ligand_path=None,
                 receptor_path=None,
                 ):
        env = "module load buster"
        ligand_fit_command = "rhofit"
        ligand_fit_args = "-m {mtz} -l {ligand} -p {receptor} -d {out_dir_path} -allclusters -use_2fofc"
        ligand_fit_args_formatted = ligand_fit_args.format(mtz=mtz_path,
                                                           ligand=ligand_path,
                                                           receptor=receptor_path,
                                                           out_dir_path=out_dir_path,
                                                           )
        self.command = "{env}; {ligand_fit_command} {args}".format(env=env,
                                                                   ligand_fit_command=ligand_fit_command,
                                                                   args=ligand_fit_args_formatted,
                                                                   )

    def __repr__(self):
        return self.command


class AutobuildingCommand:
    def __init__(self,
                 out_dir_path=None,
                 mtz_path=None,
                 ligand_path=None,
                 receptor_path=None,
                 coord=(0, 0, 0),
                 ):
        env = "module load phenix"
        ligand_fit_command = "phenix.ligandfit"
        ligand_fit_args = "data=\"{mtz}\" ligand=\"{ligand}\" model=\"{receptor}\" search_center=[{x},{y},{z}] search_dist=6"
        ligand_fit_args_formatted = ligand_fit_args.format(mtz=mtz_path,
                                                           ligand=ligand_path,
                                                           receptor=receptor_path,
                                                           x=coord[0],
                                                           y=coord[1],
                                                           z=coord[2],
                                                           )
        self.command = "{env}; cd {out_dir_path}; {ligand_fit_command} {args}".format(env=env,
                                                                                      out_dir_path=out_dir_path,
                                                                                      ligand_fit_command=ligand_fit_command,
                                                                                      args=ligand_fit_args_formatted,
                                                                                      )

    def __repr__(self):
        return self.command


def execute(command):
    # print("\t\t: {}".format(command))
    submit_proc = subprocess.Popen(str(command),
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   )
    stdout, stderr = submit_proc.communicate()
    return str(command), stdout, stderr


def event_map_to_mtz(event_map_path: Path,
                     output_path,
                     resolution,
                     map2sf_path: Path = "/dls/science/groups/i04-1/conor_dev/pandda_science/autobuilding_paper/map2sf.py",
                     ):
    command = "module load gcc/4.9.3; source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_clipper_no_mkl; python {map2sf_path} -i {event_map_path} -o {output_path} -r {resolution}"
    formatted_command = command.format(map2sf_path=map2sf_path,
                                       event_map_path=event_map_path,
                                       output_path=output_path,
                                       resolution=resolution,
                                       )
    # print(formatted_command)

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    return formatted_command, stdout, stderr


def event_map_to_mtz_dep(event_map_path: Path,
                         output_path,
                         resolution,
                         col_f="FWT",
                         col_ph="PHWT",
                         gemmi_path: Path = "/dls/science/groups/i04-1/conor_dev/gemmi/gemmi",
                         ):
    command = "module load gcc/4.9.3; source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_clipper_no_mkl; {gemmi_path} map2sf {event_map_path} {output_path} {col_f} {col_ph} --dmin={resolution}"
    formatted_command = command.format(gemmi_path=gemmi_path,
                                       event_map_path=event_map_path,
                                       output_path=output_path,
                                       col_f=col_f,
                                       col_ph=col_ph,
                                       resolution=resolution,
                                       )
    # print(formatted_command)

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    return formatted_command, stdout, stderr


def write_autobuild_log(formatted_command, stdout, stderr, autobuilding_log_path):
    with open(str(autobuilding_log_path), "w") as f:
        f.write("===Command===\n")
        f.write("{}\n".format(formatted_command))
        f.write("===Stdout===\n")
        f.write("{}\n".format(stdout))
        f.write("===Stderr===\n")
        f.write("{}\n".format(stderr))


def elbow(autobuilding_dir, ligand_smiles_path):
    command = "module load phenix; cd {autobuilding_dir}; phenix.elbow {ligand_smiles_path} --output=\"{out}\""
    formatted_command = command.format(autobuilding_dir=autobuilding_dir,
                                       ligand_smiles_path=ligand_smiles_path,
                                       out="autobuilding_ligand",
                                       )

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    return stdout, stderr


def get_ligand_smiles(pandda_event_dir):
    compound_dir = pandda_event_dir / "ligand_files"

    ligand_pdbs = list(compound_dir.glob("*.pdb"))
    ligand_pdb_strings = [str(ligand_path) for ligand_path in ligand_pdbs if ligand_path.name != "tmp.pdb"]
    if len(ligand_pdb_strings) > 0:
        shortest_ligand_path = min(ligand_pdb_strings,
                                   key=len,
                                   )
        return Path(shortest_ligand_path)

    smiles_paths = compound_dir.glob("*.smiles")
    smiles_paths_list = list(smiles_paths)

    if len(smiles_paths_list) > 0:
        return Path(min([str(ligand_path) for ligand_path in smiles_paths_list if ligand_path.name != "tmp.smiles"],
                        key=len)
                    )

    else:
        raise Exception("No smiles found! Smiles list is: {}".format(smiles_paths_list))


def get_ligand_mean_coords(residue):
    coords = []
    for atom in residue:
        pos = atom.pos
        coords.append([pos.x, pos.y, pos.z])

    coords_array = np.array(coords)
    mean_coords = np.mean(coords_array,
                          axis=0,
                          )
    return mean_coords


def get_ligand_distance(ligand, coords):
    ligand_mean_coords = get_ligand_mean_coords(ligand)
    distance = np.linalg.norm(ligand_mean_coords - coords)
    return distance


def remove_residue(chain, ligand):
    del chain[str(ligand.seqid)]


def strip_protein(initial_receptor_path,
                  coords,
                  receptor_path,
                  ):
    # Load protein
    structure = gemmi.read_structure(str(initial_receptor_path))
    receptor = structure[0]

    # Strip nearby residues
    remove_ids = []
    for chain in receptor:
        ligands = chain.get_ligands()
        for ligand in ligands:
            distance = get_ligand_distance(ligand, coords)
            if distance < 10:
                print("\t\tWill strip res {}. Mean distance {} from event!".format(ligand, distance))
                remove_ids.append(str(ligand.seqid))

    for chain in receptor:
        deletions = 0
        for i, residue in enumerate(chain):
            if str(residue.seqid) in remove_ids:
                print("\t\tStripping {}".format(residue))
                del chain[i - deletions]
                deletions = deletions + 1

    # Save
    structure.write_pdb(str(receptor_path))


# def quick_refine(initial_event_mtz_path,
#                      intial_receptor_path,
#                      ):
#
#

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


@dataclasses.dataclass()
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


@dataclasses.dataclass()
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


@dataclasses.dataclass()
class Reflections:
    reflections: typing.Dict[HKL, Reflection]

    @staticmethod
    def from_array(array):
        reflections = {}
        for row in array:
            reflection = Reflection.from_row(row)
            reflections[reflection.hkl] = reflection

        return Reflections(reflections)

    def __getitem__(self, item):
        return self.reflections[item]

    def __iter__(self):
        for hkl in self.reflections:
            yield hkl

    def to_array(self):
        rows = []
        for hkl in self.reflections:
            hkl_array = np.array(hkl.to_list())
            data = self.reflections[hkl].data

            row = np.hstack([hkl_array, data])

            rows.append(row)

        array = np.vstack(rows)

        return array


def phase_graft(initial_mtz_path,
                event_mtz_path,
                out_path,
                ):
    print("\tGrafting phases from {} to {}".format(str(event_mtz_path),
                                                   str(initial_mtz_path),
                                                   ),
          )

    initial_mtz = gemmi.read_mtz_file(str(initial_mtz_path))
    # print("\tInitial mtz spacegroup: {}".format(initial_mtz.spacegroup))
    event_mtz = gemmi.read_mtz_file(str(event_mtz_path))
    # print("\tEvent mtz spacegroup: {}".format(event_mtz.spacegroup))

    initial_mtz_data = np.array(initial_mtz, copy=False)
    # print("\tShape of initial array is {}".format(initial_mtz_data.shape))
    event_mtz_data = np.array(event_mtz, copy=False)
    # print("\tShape of event array is {}".format(event_mtz_data.shape))

    initial_reflections: Reflections = Reflections.from_array(initial_mtz_data)
    event_reflections: Reflections = Reflections.from_array(event_mtz_data)

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
        # print("\t\tasu reflection is {}".format(asu_hkl))
        if asu_hkl.is_000():
            # print("\t\t\tReflection 000 {}".format(event_reflection))
            data = np.zeros(len(list(initial_reflections.reflections.values())[0].data))
            # print("\t\t\t{}".format(data.shape))
            new_reflection = Reflection(hkl, data)

        else:
            initial_reflection: Reflection = initial_reflections[asu_hkl]

            new_reflection = Reflection(hkl, np.copy(initial_reflection.data))

        new_reflection.data[initial_mtz_fwt_index - 3] = event_reflection.data[event_mtz_fwt_index - 3]
        new_reflection.data[initial_mtz_phwt_index - 3] = event_reflection.data[event_mtz_phwt_index - 3]
        # new_reflection.data[initial_mtz_fo_index - 3] = 0.0
        # new_reflection.data[initial_mtz_fc_index - 3] = 0.0
        # new_reflection.data[initial_mtz_phc_index - 3] = 0.0
        # new_reflection.data[fom_index - 3] = 0.0
        # new_reflection.data[initial_mtz_r_index - 3] = 0.0
        #
        # new_reflection.data[initial_mtz_ls_fc_all_index - 3] = 0.0
        # new_reflection.data[initial_mtz_ls_phc_all_index - 3] = 0.0
        # new_reflection.data[initial_mtz_fc_all_index - 3] = 0.0
        # new_reflection.data[initial_mtz_phc_all_index - 3] = 0.0
        # new_reflection.data[initial_mtz_delfwt_index - 3] = 0.0
        # new_reflection.data[initial_mtz_phdelwt_index - 3] = 0.0
        #
        # new_reflection.data[initial_mtz_sigf_index - 3] = 0.0

        # event_reflection = event_reflections[hkl]
        #
        # asu_hkl = HKL.from_list(initial_asu.to_asu(hkl.to_list(), operations, ))
        # if asu_hkl.is_000():
        #     print("\t\t{}".format([hkl, asu_hkl]))
        #
        # print("#### zeros array, normal array")
        # print(np.zeros(len(list(initial_reflections.reflections.values())[0].data)).shape)
        # print(list(initial_reflections.reflections.values())[0].data.shape)
        #
        # data = np.zeros(len(list(initial_reflections.reflections.values())[0].data))
        # new_reflection = Reflection(hkl, data)
        #
        # new_reflection.data[initial_mtz_fwt_index - 3] = event_reflection.data[event_mtz_fwt_index - 3]
        # new_reflection.data[initial_mtz_phwt_index - 3] = event_reflection.data[event_mtz_phwt_index - 3]

        new_reflections[hkl] = new_reflection

    print("\tFinished iterating reflections")

    new_array = Reflections(new_reflections).to_array()
    # print("\tShape of new array is {}".format(new_array.shape))

    initial_mtz.spacegroup = event_mtz.spacegroup
    initial_mtz.set_data(new_array)

    np.core.arrayprint._line_width = 240
    # print([initial_mtz_fwt_index, initial_mtz_phwt_index, event_mtz_fwt_index, event_mtz_phwt_index])
    # print(new_array.shape)
    # print(new_array[-5:-1,:])
    # print(np.array(initial_mtz)[-5:-1,:])
    # print(np.array(event_mtz)[-5:-1,:])

    print("\tWriting new reflections to {}".format(str(out_path)))
    initial_mtz.write_to_file(str(out_path))

    # initial_mtz_file = gemmi.read_mtz_file(str(out_path))
    # initial_mtz_array = np.array(initial_mtz_file, copy=False)
    # print(np.array(initial_mtz_array)[-5:-1,:])


def phase_graft_dep(initial_mtz_path,
                    event_mtz_path,
                    out_path,
                    ):
    intial_mtz = gemmi.read_mtz_file(str(initial_mtz_path))
    event_mtz = gemmi.read_mtz_file(str(event_mtz_path))

    array_to_index_map = array_to_index(intial_mtz)
    index_to_array_map = index_to_array(event_mtz)

    initial_mtz_data = np.array(intial_mtz, copy=False)
    event_mtz_data = np.array(event_mtz, copy=False)
    # print(initial_mtz_data.shape)
    # print(event_mtz_data.shape)

    # FWT
    initial_mtz_fwt = intial_mtz.column_with_label('FWT')
    # initial_mtz_fwt_index = initial_mtz_fwt.dataset_id
    initial_mtz_fwt_index = intial_mtz.column_labels().index("FWT")

    event_mtz_fwt = event_mtz.column_with_label('FWT')
    event_mtz_fwt_index = event_mtz.column_labels().index("FWT")

    # print("\t{}, {}".format(initial_mtz_data.shape, event_mtz_data.shape))
    # print(list(array_to_index_map.keys())[:10])
    # print(list(index_to_array_map.keys())[:10])

    skipped = 0
    for intial_array in range(initial_mtz_data.shape[0]):
        try:
            index = array_to_index_map[intial_array]
            event_array = index_to_array_map[index]
            initial_mtz_data[intial_array, initial_mtz_fwt_index] = event_mtz_data[event_array, event_mtz_fwt_index]
        except Exception as e:
            skipped = skipped + 1
            initial_mtz_data[intial_array, initial_mtz_fwt_index] = 0
    intial_mtz.set_data(initial_mtz_data)
    print("\tSkipped {} reflections".format(skipped))

    # PHWT
    initial_mtz_phwt = intial_mtz.column_with_label('PHWT')
    # initial_mtz_phwt_index = initial_mtz_phwt.dataset_id
    initial_mtz_phwt_index = intial_mtz.column_labels().index("PHWT")

    event_mtz_phwt = event_mtz.column_with_label('PHWT')
    # event_mtz_phwt_index = event_mtz_phwt.dataset_id
    event_mtz_phwt_index = event_mtz.column_labels().index("PHWT")

    skipped = 0
    for intial_array in range(initial_mtz_data.shape[0]):
        try:
            index = array_to_index_map[intial_array]
            event_array = index_to_array_map[index]
            initial_mtz_data[intial_array, initial_mtz_phwt_index] = event_mtz_data[event_array, event_mtz_phwt_index]
        except Exception as e:
            skipped = skipped + 1
            initial_mtz_data[intial_array, initial_mtz_phwt_index] = 0
    intial_mtz.set_data(initial_mtz_data)
    print("\tCopied FWT from {} to {}".format(event_mtz_fwt_index, initial_mtz_fwt_index))
    print("\tCopied PHWT from {} to {}".format(event_mtz_phwt_index, initial_mtz_phwt_index))
    print("\tSkipper {} reflections".format(skipped))

    intial_mtz.spacegroup = event_mtz.spacegroup

    intial_mtz.write_to_file(str(out_path))


def try_remove(ligand_path):
    if ligand_path.exists():
        os.remove(str(ligand_path))


def autobuild_event(event):
    try:

        # Event map mtz
        print("\tMaking event map mtz...")
        initial_event_mtz_path = event.pandda_event_dir / "{}_{}.mtz".format(event.dtag, event.event_idx)
        try:
            os.remove(str(initial_event_mtz_path))
        except:
            pass

        formatted_command, stdout, stderr = event_map_to_mtz(event.event_map_path,
                                                             initial_event_mtz_path,
                                                             event.analysed_resolution,
                                                             )
        print("\tMtz command: {}".format(formatted_command))
        event_mtz_log = event.pandda_event_dir / "event_mtz_log.txt"
        write_autobuild_log(formatted_command, stdout, stderr, event_mtz_log)

        # Ligand cif
        print("\tMaking ligand cif...")
        ligand_path = event.pandda_event_dir / "autobuilding_ligand.cif"
        try_remove(ligand_path)
        ligand_smiles_path = get_ligand_smiles(event.pandda_event_dir)
        print(ligand_smiles_path)
        # if not ligand_path.exists():
        elbow(event.pandda_event_dir,
              ligand_smiles_path,
              )

        # Stripped protein
        print("\tStripping ligands near event...")
        intial_receptor_path = event.pandda_event_dir / "receptor_{}.pdb".format(event.event_idx)
        # if not intial_receptor_path.exists():
        strip_protein(event.receptor_path,
                      event.coords,
                      intial_receptor_path,
                      )

        # Quick refine
        event_mtz_path = event.pandda_event_dir / "grafted_{}.mtz".format(event.event_idx)
        try:
            os.remove(str(event_mtz_path))
        except:
            pass
        # if not event_mtz_path.exists():
        phase_graft(event.initial_mtz_path,
                    initial_event_mtz_path,
                    event_mtz_path,
                    )

        if not event_mtz_path.exists():
            raise Exception("Could not find event mtz after attempting generation: {}".format(event_mtz_path))

        if not ligand_path.exists():
            raise Exception("Could not find ligand cif path after attempting generation: {}".format(event_mtz_path))

        # if not receptor_path.exists():
        #     raise Exception("Could not find event receptor path after attempting generation: {}".format(event_mtz_path))

        out_dir_path = event.pandda_event_dir / "rhofit_{}".format(event.event_idx)

        try:
            shutil.rmtree(str(out_dir_path))
        except:
            pass

        # os.mkdir(str(out_dir_path))

        # autobuilding_command = AutobuildingCommand(out_dir_path=out_dir_path,
        #                                            mtz_path=event_mtz_path,
        #                                            ligand_path=event.ligand_path,
        #                                            receptor_path=receptor_path,
        #                                            coord=event.coords,
        #                                            )
        #
        # formatted_command, stdout, stderr = execute(autobuilding_command)
        #
        # autobuilding_log_path = out_dir_path / "pandda_autobuild_log.txt"
        # write_autobuild_log(formatted_command, stdout, stderr, autobuilding_log_path)

        print("\tAutobuilding...")
        # if (out_dir_path / "results.txt").exists():
        autobuilding_command = AutobuildingCommandRhofit(out_dir_path=out_dir_path,
                                                         mtz_path=event_mtz_path,
                                                         ligand_path=ligand_path,
                                                         receptor_path=intial_receptor_path,
                                                         )
        print("\t\tCommand: {}".format(str(autobuilding_command)))
        formatted_command, stdout, stderr = execute(autobuilding_command)

        try:
            print("\tProcessing autobuilding results...")
            autobuilding_log_path = out_dir_path / "pandda_autobuild_log.txt"
            write_autobuild_log(formatted_command, stdout, stderr, autobuilding_log_path)
            result = AutobuildingResultRhofit.from_output(event,
                                                          stdout,
                                                          stderr,
                                                          )
        except:
            result = AutobuildingResultRhofit.null_result(event)

    except Exception as e:
        result = AutobuildingResultRhofit.null_result(event)

    return result


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
    def __init__(self):
        input_pandda_dir = Option("i", "input_pandda_dir", required=True)
        overwrite = Option("o", "overwrite", required=True)
        pandda_version = Option("p", "pandda_version", required=True)

        options = [input_pandda_dir, overwrite, pandda_version]

        args = self.get_args(options)

        self.input_pandda_dir = Path(input_pandda_dir(args))
        self.overwrite = bool(args.overwrite)
        self.pandda_version = int(args.pandda_version)

    def get_args(self, options):
        parser = argparse.ArgumentParser()
        for option in options:
            parser.add_argument("-{}".format(option.short),
                                "--{}".format(option.long),
                                help=option.help,
                                required=option.required,
                                default=option.default,
                                )
        return parser.parse_args()


class PanDDAFilesystemModel:
    def __init__(self, pandda_root_dir):
        self.pandda_root_dir = pandda_root_dir

        self.pandda_analyse_dir = pandda_root_dir / "analyses"
        self.pandda_inspect_events_path = self.pandda_analyse_dir / "pandda_analyse_events.csv"
        self.autobuilding_results_table = self.pandda_analyse_dir / "autobuilding_results.csv"

        self.pandda_processed_datasets_dir = pandda_root_dir / "processed_datasets"
        self.pandda_processed_datasets_dirs = list(self.pandda_analyse_dir.glob("*"))


def map_parallel(f, datasets):
    results = joblib.Parallel(n_jobs=20,
                              verbose=50,
                              )(joblib.delayed(f)(dataset)
                                for dataset
                                in datasets)

    # results = []
    # for dataset in datasets:
    #     result = f(dataset)
    #     print(result)
    #     print(result.rscc)
    #     results.append(result)

    return results


# def get_datasets(fs: PanDDAFilesystemModel):
#     datasets = []
#     for processed_dataset_dir in fs.pandda_processed_datasets_dirs:
#         dataset = Dataset.from_dir(processed_dataset_dir)
#         datasets.append(dataset)
#
#     return datasets

class Event:
    def __init__(self,
                 pandda_event_dir,
                 dtag,
                 event_idx,
                 event_map_path,
                 initial_mtz_path,
                 ligand_path,
                 receptor_path,
                 coords,
                 analysed_resolution,
                 ):
        self.dtag = dtag
        self.event_idx = event_idx
        self.pandda_event_dir = pandda_event_dir
        self.event_map_path = event_map_path
        self.initial_mtz_path = initial_mtz_path
        self.ligand_path = ligand_path
        self.receptor_path = receptor_path
        self.coords = coords
        self.analysed_resolution = analysed_resolution


def get_event_table(path):
    return pd.read_csv(str(path))


def get_event_map_path(pandda_event_dir, dtag, event_idx, occupancy, pandda_version=2):
    # PanDDA 1
    if pandda_version == 2:
        event_map_path = pandda_event_dir / "{}-event_{}_1-BDC_{}_map.ccp4".format(dtag,
                                                                                   event_idx,
                                                                                   occupancy,
                                                                                   )
    # PanDDA 2
    if pandda_version == 1:
        event_map_path = pandda_event_dir / "{}-event_{}_1-BDC_{}_map.native.ccp4".format(dtag,
                                                                                          event_idx,
                                                                                          occupancy,
                                                                                          )
    return event_map_path


def get_ligand_path(pandda_event_dir):
    # print("\t\t{}".format(pandda_event_dir))

    event_dir = pandda_event_dir
    #
    # ligands = list((event_dir / "ligand_files").glob("*.pdb"))
    # ligand_strings = [str(ligand_path) for ligand_path in ligands if ligand_path.name != "tmp.pdb"]
    #
    # if len(ligand_strings) == 0:
    #     return None
    #
    # ligand_pdb_path: Path = Path(min(ligand_strings,
    #                                  key=len,
    #                                  )
    #                              )
    # return ligand_pdb_path

    ligands = list((event_dir / "ligand_files").glob("*.smiles"))
    ligand_strings = [str(ligand_path) for ligand_path in ligands if ligand_path.name != "tmp.smiles"]

    if len(ligand_strings) > 0:
        ligand_smiles_path: Path = Path(min(ligand_strings,
                                            key=len,
                                            )
                                        )

        return ligand_smiles_path

    ligand_pdbs = list((event_dir / "ligand_files").glob("*.pdb"))
    ligand_pdb_strings = [str(ligand_path) for ligand_path in ligand_pdbs if ligand_path.name != "tmp.pdb"]
    if len(ligand_pdb_strings) > 0:
        shortest_ligand_path = min(ligand_pdb_strings,
                                   key=len,
                                   )
        return Path(shortest_ligand_path)

    else:
        return None


def get_receptor_path(pandda_event_dir, dtag):
    receptor_path = pandda_event_dir / "{}-pandda-input.pdb".format(dtag)
    return receptor_path


def get_coords(row):
    return (row["x"], row["y"], row["z"])


def get_analyed_resolution(row):
    return row["analysed_resolution"]


def get_events(event_table, fs, pandda_version):
    events = []
    for index, row in event_table.iterrows():
        dtag = row["dtag"]
        event_idx = row["event_idx"]
        occupancy = row["1-BDC"]
        pandda_event_dir = fs.pandda_processed_datasets_dir / "{}".format(dtag)
        event_map_path = get_event_map_path(pandda_event_dir,
                                            dtag, event_idx, occupancy,
                                            pandda_version,
                                            )
        ligand_path = get_ligand_path(pandda_event_dir)
        if ligand_path is None:
            print("\tCould not find smiles for {} in directory {}".format(dtag,
                                                                          pandda_event_dir,
                                                                          )
                  )
            continue

        receptor_path = get_receptor_path(pandda_event_dir, dtag)
        coords = get_coords(row)
        analysed_resolution = get_analyed_resolution(row)

        initial_mtz_path = pandda_event_dir / "{}-pandda-input.mtz".format(dtag)

        event = Event(pandda_event_dir,
                      dtag,
                      event_idx,
                      event_map_path,
                      initial_mtz_path,
                      ligand_path,
                      receptor_path,
                      coords,
                      analysed_resolution,
                      )
        events.append(event)

    return events


class AutobuildingResultRhofit:
    def __init__(self, dtag, event_idx, rscc_string, stdout, stderr):
        self.dtag = dtag
        self.event_idx = event_idx
        self.rscc = float(rscc_string)
        self.stdout = stdout
        self.stderr = stderr

    @staticmethod
    def null_result(event):
        return AutobuildingResultRhofit(event.dtag,
                                        event.event_idx,
                                        "0",
                                        "",
                                        "",
                                        )

    @staticmethod
    def from_output(event: Event, stdout, stderr):
        rhofit_dir = event.pandda_event_dir / "rhofit_{}".format(event.event_idx)
        rhofit_results_path = rhofit_dir / "results.txt"

        rscc_string = str(AutobuildingResultRhofit.parse_rhofit_log(rhofit_results_path))
        return AutobuildingResultRhofit(event.dtag, event.event_idx, rscc_string, stdout, stderr)

    @staticmethod
    def parse_rhofit_log(rhofit_results_path):
        regex = "(Hit_[^\s]+)[\s]+[^\s]+[\s]+[^\s]+[\s]+([^\s]+)"

        with open(str(rhofit_results_path), "r") as f:
            results_string = f.read()

        rscc_matches = re.findall(regex,
                                  results_string,
                                  )

        rsccs = {str(match_tuple[0]): float(match_tuple[1]) for match_tuple in rscc_matches}

        max_rscc = max(list(rsccs.values()))

        return max_rscc


class AutobuildingResult:
    def __init__(self, dtag, event_idx, rscc_string, stdout, stderr):
        self.dtag = dtag
        self.event_idx = event_idx
        self.rscc = float(rscc_string)
        self.stdout = stdout
        self.stderr = stderr

    @staticmethod
    def null_result(event):
        return AutobuildingResult(event.dtag,
                                  event.event_idx,
                                  "0",
                                  "",
                                  "",
                                  )

    @staticmethod
    def from_output(event: Event, stdout, stderr):
        event_autobuilding_dir = event.pandda_event_dir / "autobuild_event_{}".format(event.event_idx)
        event_ligandfit_dir = event_autobuilding_dir / "LigandFit_run_1_"
        autobuilding_results_file = event_ligandfit_dir / "LigandFit_summary.dat"

        with open(str(autobuilding_results_file), "r") as f:
            result_string = f.read()

        rscc_regex = "[\s]+1[\s]+[0-9\.]+[\s]+([0-9\.]+)"
        match = re.findall(rscc_regex, result_string)
        rscc_string = match[0]
        return AutobuildingResult(event.dtag, event.event_idx, rscc_string, stdout, stderr)


class ResultsTable:
    def __init__(self, builds: typing.Dict[typing.Tuple[str, int], Builds]):
        records = []
        for key, build in builds.items():
            record = {}
            record["dtag"] = build.dtag
            record["event_idx"] = build.event_idx
            record["rscc"] = max(build.build_results.values())

            records.append(record)

        self.table = pd.DataFrame(records)

    def to_csv(self, path):
        self.table.to_csv(str(path))


def get_highest_rscc_events(events,
                            results_table,
                            ):
    unique_dtags = results_table["dtag"].unique()

    events_map = {(event.dtag, event.event_idx): event for event in events}

    max_events = []
    for unique_dtag in unique_dtags:
        events_table = results_table[results_table["dtag"] == unique_dtag]
        max_rscc_label = events_table["rscc"].idxmax()
        event_row = events_table.loc[max_rscc_label]
        dtag = event_row["dtag"]
        event_idx = event_row["event_idx"]
        max_events.append(events_map[(dtag, event_idx)])

    return max_events


#
# def copy_event_to_processed_models(event: Event, fs):
#     event_autobuilding_dir = event.pandda_event_dir / "autobuild_event_{}".format(event.event_idx)
#     event_ligandfit_dir = event_autobuilding_dir / "LigandFit_run_1_"
#     event_build_path = event_ligandfit_dir / "ligand_fit_1.pdb"
#
#     initial_model_path = event_autobuilding_dir / "{}-pandda-input.pdb".format(event.dtag)
#
#     pandda_inspect_model_dir = event.pandda_event_dir / "modelled_structures"
#     pandda_inspect_model_path = pandda_inspect_model_dir / "{}-pandda-model.pdb".format(event.dtag)
#
#     initial_model = PandasPdb().read_pdb(str(initial_model_path))
#     best_autobuild_model = PandasPdb().read_psb(str(event_build_path))
#
#     initial_model.df["HETATM"] = initial_model.df["HETATM"].append(best_autobuild_model.df["HETATM"])
#
#     renumber(initial_model.df["HETATM"])
#
#     #
#     # shutil.copyfile(str(event_build_path),
#     #                 str(pandda_inspect_model_path),
#     #                 )


def save_event_model(event_model, path):
    event_model.write_pdb(str(path))


def merge_model(event, fs, build: Builds):
    # event_autobuilding_dir = event.pandda_event_dir / "autobuild_event_{}".format(event.event_idx)
    # event_ligandfit_dir = event_autobuilding_dir / "LigandFit_run_1_"
    # event_build_path = event_ligandfit_dir / "ligand_fit_1.pdb"

    event_autobuilding_dir = event.pandda_event_dir
    event_rhofit_dir = event_autobuilding_dir / "rhofit_{}".format(event.event_idx)
    # event_build_path = event_rhofit_dir / "best.pdb"
    event_build_path: Path = max(build.build_results.keys(),
                                 key=lambda x: build.build_results[x],
                                 )

    initial_model_path = event.pandda_event_dir / "{}-pandda-input.pdb".format(event.dtag)

    # initial_model = PandasPdb().read_pdb(str(initial_model_path))
    # best_autobuild_model = PandasPdb().read_psb(str(event_build_path))

    initial_model = gemmi.read_structure(str(initial_model_path))
    best_autobuild_model = gemmi.read_structure(str(event_build_path))

    # initial_model.df["HETATM"] = initial_model.df["HETATM"].append(best_autobuild_model.df["HETATM"])

    print("\tBefore adding lig there are {} chains".format(len(initial_model[0])))
    initial_model[0].add_chain(best_autobuild_model[0][0])
    print("\tAfter adding lig there are {} chains".format(len(initial_model[0])))

    return initial_model


def try_make(path):
    if not path.exists():
        os.mkdir(str(path))


def merge_models(events,
                 build_results: typing.Dict[typing.Tuple[str, int], Builds],
                 results_table,
                 fs,
                 overwrite=False,
                 ):
    results_dict = {(result.dtag, result.event_idx): result for result in build_results.values()}

    highest_rscc_events = get_highest_rscc_events(events,
                                                  results_table,
                                                  )

    print("\t\tAfter filetering duplicate events got {} events".format(len(highest_rscc_events)))

    for event in highest_rscc_events:
        if results_dict[(event.dtag, event.event_idx)].rscc == 0.0:
            print("\tNo build for event! Skipping!")
            continue
        final_model = merge_model(event, fs, build_results[(event.dtag, event.event_idx)])

        pandda_inspect_model_dir = event.pandda_event_dir / "modelled_structures"
        try_make(pandda_inspect_model_dir)
        pandda_inspect_model_path = pandda_inspect_model_dir / "{}-pandda-model.pdb".format(event.dtag)
        if (overwrite or (not pandda_inspect_model_path.exists())):
            save_event_model(final_model,
                             pandda_inspect_model_path,
                             )
        else:
            print("\tAlready has a model, skipping!")


# @dataclasses.dataclass()
# class Build:
#     build_results: typing.Dict[]


@dataclasses.dataclass()
class Builds:
    dtag: str
    event_idx: int
    build_results: typing.Dict[Path, float]
    rscc: float

    @staticmethod
    def from_event(event: Event):
        dtag: str = event.dtag
        event_idx: int = event.event_idx
        event_dir: Path = event.pandda_event_dir

        rhofit_rsccs: typing.Dict[Path, float] = Builds.parse_rhofit_results(event_dir, event_idx)

        rscc = max(rhofit_rsccs.values())

        return Builds(dtag, event_idx, rhofit_rsccs, rscc)

    @staticmethod
    def parse_rhofit_results(event_dir: Path, event_idx: int):
        rhotfit_dir: Path = event_dir / "rhofit_{}".format(event_idx)
        rhofit_results_file: Path = rhotfit_dir / "results.txt"

        regex = "(Hit_[^\s]+)[\s]+[^\s]+[\s]+[^\s]+[\s]+([^\s]+)"

        with open(str(rhofit_results_file), "r") as f:
            results_string = f.read()

        rscc_matches = re.findall(regex,
                                  results_string,
                                  )

        rsccs = {rhotfit_dir / str(match_tuple[0]): float(match_tuple[1]) for match_tuple in rscc_matches}

        return rsccs


# @dataclasses.dataclass()
# class BuildResults:
#     build_results: typing.Dict[typing.Tuple[str, str, str], BuildResult]
#
#     @staticmethod
#     def from_events(events: typing.List[Event]):
#         build_results = []
#         for event in events:
#             build_result = Builds.from_event(event)
#             build_results.append(build_result)
#
#         build_results_dict = {{build_result.}}


def main():
    config = Config()

    print("Building I/O model...")
    fs = PanDDAFilesystemModel(config.input_pandda_dir)
    print("\tFound {} dataset dirs".format(len(fs.pandda_processed_datasets_dirs)))

    print("Geting event table...")
    event_table = get_event_table(fs.pandda_inspect_events_path)
    print("\tFound {} PanDDA events".format(len(event_table)))

    print("Getting event models...")
    events = get_events(event_table,
                        fs,
                        config.pandda_version,
                        )
    print("\tGot models of {} events".format(len(events)))

    print("Autobuilding...")
    # autobuilding_results = map_parallel(autobuild_event,
    #                                     events,
    #                                     )
    # print("\tAutobuilt {} events".format(len(autobuilding_results)))
    # for result in autobuilding_results: print("\t{} {} : RSCC: {}".format(result.dtag, result.event_idx, result.rscc))

    build_results: typing.Dict[typing.Tuple[str, int], Builds] = {}
    for event in events:
        try:
            build_result = Builds.from_event(event)
            build_results[(event.dtag, event.event_idx)] = build_result
        except Exception as e:
            continue

    print("Making autobuilding results table...")
    results_table = ResultsTable(build_results)
    print("\tMade autobuilding resutls table")

    print("Merging best models")
    merge_models(events,
                 build_results,
                 results_table.table,
                 fs,
                 overwrite=config.overwrite,
                 )

    print("Outputing autobuilding results table...")
    results_table.to_csv(fs.autobuilding_results_table)
    print("\tOutput autobuilding results table")


if __name__ == "__main__":
    main()
