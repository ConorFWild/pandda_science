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
        ligand_fit_args = "-m {mtz} -l {ligand} -p {receptor} -d {out_dir_path} -allclusters"
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
                                       out="ligand",
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

    smiles_paths = compound_dir.glob("*.smiles")

    smiles_paths_list = list(smiles_paths)

    if len(smiles_paths_list) == 0:

        raise Exception("No smiles found! Smiles list is: {}".format(smiles_paths_list))

    else:
        return smiles_paths_list[0]


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


def phase_graft(initial_mtz_path,
                event_mtz_path,
                out_path,
                ):
    intial_mtz = gemmi.read_mtz_file(str(initial_mtz_path))
    event_mtz = gemmi.read_mtz_file(str(event_mtz_path))

    array_to_index_map = array_to_index(intial_mtz)
    index_to_array_map = index_to_array(event_mtz)

    # FWT
    initial_mtz_fwt = intial_mtz.column_with_label('DELFWT')
    initial_mtz_fwt_index = initial_mtz_fwt.dataset_id
    initial_mtz_data = np.array(intial_mtz, copy=False)
    print("\t{}".format(initial_mtz_data.shape))

    event_mtz_fwt = event_mtz.column_with_label('FWT')
    event_mtz_fwt_index = event_mtz_fwt.dataset_id
    event_mtz_data = np.array(event_mtz, copy=False)
    print("\t{}".format(initial_mtz_data.shape))

    for intial_array in range(initial_mtz_data.shape[0]):
        index = array_to_index_map[intial_array]
        event_array = index_to_array_map[index]
        initial_mtz_data[intial_array, initial_mtz_fwt_index] = event_mtz_data[event_array, event_mtz_fwt_index]
    intial_mtz.set_data(initial_mtz_data)

    # PHWT
    initial_mtz_fwt = intial_mtz.column_with_label('DELPHWT')
    initial_mtz_fwt_index = initial_mtz_fwt.dataset_id
    initial_mtz_data = np.array(intial_mtz, copy=False)

    event_mtz_fwt = event_mtz.column_with_label('PHWT')
    event_mtz_fwt_index = event_mtz_fwt.dataset_id
    event_mtz_data = np.array(event_mtz, copy=False)

    for intial_array in range(initial_mtz_data.shape[0]):
        index = array_to_index_map[intial_array]
        event_array = index_to_array_map[index]
        initial_mtz_data[intial_array, initial_mtz_fwt_index] = event_mtz_data[event_array, event_mtz_fwt_index]
    intial_mtz.set_data(initial_mtz_data)

    intial_mtz.write_to_file(str(out_path))


def autobuild_event(event):
    # Event map mtz
    print("\tMaking event map mtz...")
    initial_event_mtz_path = event.pandda_event_dir / "{}_{}.mtz".format(event.dtag, event.event_idx)

    formatted_command, stdout, stderr = event_map_to_mtz(event.event_map_path,
                                                         initial_event_mtz_path,
                                                         event.analysed_resolution,
                                                         )
    event_mtz_log = event.pandda_event_dir / "event_mtz_log.txt"
    write_autobuild_log(formatted_command, stdout, stderr, event_mtz_log)

    # Ligand cif
    print("\tMaking ligand cif...")
    ligand_path = event.pandda_event_dir / "ligand.cif"
    ligand_smiles_path = get_ligand_smiles(event.pandda_event_dir)
    if not ligand_path.exists():
        elbow(event.pandda_event_dir,
              ligand_smiles_path,
              )

    # Stripped protein
    print("\tStripping ligands near event...")
    intial_receptor_path = event.pandda_event_dir / "receptor_{}.pdb".format(event.event_idx)
    if not intial_receptor_path.exists():
        strip_protein(event.receptor_path,
                      event.coords,
                      intial_receptor_path,
                      )

    # Quick refine
    event_mtz_path = event.pandda_event_dir / "{}.mtz".format(event.event_idx)
    if not event_mtz_path.exists():
        phase_graft(event.initial_mtz_path,
                    initial_event_mtz_path,
                    intial_receptor_path,
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
    autobuilding_command = AutobuildingCommandRhofit(out_dir_path=out_dir_path,
                                                     mtz_path=event_mtz_path,
                                                     ligand_path=ligand_path,
                                                     receptor_path=intial_receptor_path,
                                                     )
    print("\t\tCommand: {}".format(str(autobuilding_command)))
    formatted_command, stdout, stderr = execute(autobuilding_command)

    print("\tProcessing autobuilding results...")
    autobuilding_log_path = out_dir_path / "pandda_autobuild_log.txt"
    write_autobuild_log(formatted_command, stdout, stderr, autobuilding_log_path)

    try:
        result = AutobuildingResultRhofit.from_output(event,
                                                      stdout,
                                                      stderr,
                                                      )
    except:
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
        options = [input_pandda_dir]

        args = self.get_args(options)

        self.input_pandda_dir = Path(input_pandda_dir(args))

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
        self.pandda_inspect_events_path = self.pandda_analyse_dir / "pandda_inspect_events.csv"
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


def get_event_map_path(pandda_event_dir, dtag, event_idx, occupancy):
    # PanDDA 1
    # event_map_path = pandda_event_dir / "{}-event_{}_1-BDC_{}_map.native.ccp4".format(dtag,
    #                                                                                   event_idx,
    #                                                                                   occupancy,
    #                                                                                   )
    # PanDDA 2
    event_map_path = pandda_event_dir / "{}-event_{}_1-BDC_{}_map.ccp4".format(dtag,
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

    if len(ligand_strings) == 0:
        return None

    ligand_smiles_path: Path = Path(min(ligand_strings,
                                        key=len,
                                        )
                                    )

    return ligand_smiles_path


def get_receptor_path(pandda_event_dir, dtag):
    receptor_path = pandda_event_dir / "{}-pandda-input.pdb".format(dtag)
    return receptor_path


def get_coords(row):
    return (row["x"], row["y"], row["z"])


def get_analyed_resolution(row):
    return row["analysed_resolution"]


def get_events(event_table, fs):
    events = []
    for index, row in event_table.iterrows():
        dtag = row["dtag"]
        event_idx = row["event_idx"]
        occupancy = row["1-BDC"]
        pandda_event_dir = fs.pandda_processed_datasets_dir / "{}".format(dtag)
        event_map_path = get_event_map_path(pandda_event_dir,
                                            dtag, event_idx, occupancy
                                            )
        ligand_path = get_ligand_path(pandda_event_dir)
        if ligand_path is None:
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
    def __init__(self, results):
        records = []
        for result in results:
            record = {}
            record["dtag"] = result.dtag
            record["event_idx"] = result.event_idx
            record["rscc"] = result.rscc

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


def merge_model(event, fs):
    # event_autobuilding_dir = event.pandda_event_dir / "autobuild_event_{}".format(event.event_idx)
    # event_ligandfit_dir = event_autobuilding_dir / "LigandFit_run_1_"
    # event_build_path = event_ligandfit_dir / "ligand_fit_1.pdb"

    event_autobuilding_dir = event.pandda_event_dir
    event_rhofit_dir = event_autobuilding_dir / "rhofit_{}".format(event.event_idx)
    event_build_path = event_rhofit_dir / "best.pdb"

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


def merge_models(events,
                 autobuilding_results,
                 results_table,
                 fs,
                 ):
    results_dict = {(result.dtag, result.event_idx): result for result in autobuilding_results}

    highest_rscc_events = get_highest_rscc_events(events,
                                                  results_table,
                                                  )

    print("\t\tAfter filetering duplicate events got {} events".format(len(highest_rscc_events)))

    for event in highest_rscc_events:
        if results_dict[(event.dtag, event.event_idx)].rscc == 0.0:
            print("\tNo build for event! Skipping!")
            continue
        final_model = merge_model(event, fs)

        pandda_inspect_model_dir = event.pandda_event_dir / "modelled_structures"
        pandda_inspect_model_path = pandda_inspect_model_dir / "{}-pandda-model.pdb".format(event.dtag)
        if not pandda_inspect_model_path.exists():
            save_event_model(final_model,
                             pandda_inspect_model_path,
                             )
        else:
            print("\tAlready has a model, skipping!")


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
                        )
    print("\tGot models of {} events".format(len(events)))

    print("Autobuilding...")
    autobuilding_results = map_parallel(autobuild_event,
                                        events,
                                        )
    print("\tAutobuilt {} events".format(len(autobuilding_results)))
    for result in autobuilding_results: print("\t{} {} : RSCC: {}".format(result.dtag, result.event_idx, result.rscc))

    print("Making autobuilding results table...")
    results_table = ResultsTable(autobuilding_results)
    print("\tMade autobuilding resutls table")

    print("Merging best models")
    merge_models(events,
                 autobuilding_results,
                 results_table.table,
                 fs,
                 )

    print("Outputing autobuilding results table...")
    results_table.to_csv(fs.autobuilding_results_table)
    print("\tOutput autobuilding results table")


if __name__ == "__main__":
    main()
