from typing import NamedTuple, List
import os
import time
import subprocess
import shutil
import re
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import joblib

import luigi

from pandda_types.data import Event
from pandda_autobuilding.coarse import coarse_build
from pandda_autobuilding.strip import strip_receptor_waters


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--event_table_path",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-o", "--out_dir_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    args = parser.parse_args()

    return args


class Config(NamedTuple):
    out_dir_path: Path
    event_table_path: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir_path),
                    event_table_path=Path(args.event_table_path),
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path
        self.parallel_pandda_table_path = out_dir_path / "parallel_pandda_table.csv"

    def attempt_mkdir(self, path: Path):
        try:
            os.mkdir(str(path))
        except Exception as e:
            print(e)

    def attempt_remove(self, path: Path):
        try:
            shutil.rmtree(path,
                          ignore_errors=True,
                          )
        except Exception as e:
            print(e)

    def make(self, overwrite=False):
        # Overwrite old results as appropriate
        if overwrite is True:
            self.attempt_remove(self.out_dir_path)

        # Make output dirs
        self.attempt_mkdir(self.out_dir_path)


def setup_output_directory(path: Path, overwrite: bool = False):
    output: Output = Output(path)
    output.make(overwrite)
    return output


class QSub:
    def __init__(self,
                 command,
                 submit_script_path,
                 queue="low.q",
                 cores=1,
                 m_mem_free=10,
                 h_vmem=20,
                 ):
        self.command = command
        self.submit_script_path = submit_script_path
        self.queue = queue
        self.cores = cores
        self.m_mem_free = m_mem_free
        self.h_vmem = h_vmem

        with open(str(submit_script_path), "w") as f:
            f.write(command)

        chmod_proc = subprocess.Popen("chmod 777 {}".format(submit_script_path),
                                      shell=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      )
        chmod_proc.communicate()

        qsub_command = "qsub -q {queue} -pe smp {cores} -l m_mem_free={m_mem_free}G,h_vmem={h_vmem}G {submit_script_path}"
        self.qsub_command = qsub_command.format(queue=self.queue,
                                                cores=self.cores,
                                                m_mem_free=self.m_mem_free,
                                                h_vmem=self.h_vmem,
                                                submit_script_path=self.submit_script_path,
                                                )

    def __call__(self):
        print("\tCommand is: {}".format(self.command))
        print("\tQsub command is: {}".format(self.qsub_command))
        submit_proc = subprocess.Popen(self.qsub_command,
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       )
        stdout, stderr = submit_proc.communicate()

        proc_number = self.parse_process_number(str(stdout))

        time.sleep(10)

        while not self.is_finished(proc_number):
            time.sleep(10)

    def parse_process_number(self, string):
        regex = "[0-9]+"
        m = re.search(regex,
                      string,
                      )
        return m.group(0)

    def is_finished(self, proc_number):
        stat_proc = subprocess.Popen("qstat",
                                     shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     )
        stdout, stderr = stat_proc.communicate()

        if re.search(proc_number, str(stdout)):
            return False
        else:
            return True


class AutobuildStatus:
    time: float
    success: bool
    result_model_path: List[Path]

    def __init__(self, output_dir, time_taken):
        self.output_dir = output_dir
        self.time = time_taken

        ligand_fit_dir = output_dir / "LigandFit_run_1_"
        ligand_fit_pdb_path = ligand_fit_dir / "ligand_fit_1.pdb"

        if ligand_fit_pdb_path.exists():
            self.success = True
            self.result_model_path = [ligand_fit_pdb_path]
        else:
            self.success = False
            self.result_model_path = []

    def to_json(self):
        record = {}
        record["time"] = self.time
        record["success"] = self.success
        record["result_model_path"] = [str(result_path) for result_path in self.result_model_path]

        json_string = json.dumps(record)
        with open(str(self.output_dir / "task_results.json"), "w") as f:
            f.write(json_string)


class AutobuildStatusRhofit:
    def __init__(self, output_dir, time_taken):
        self.output_dir = output_dir
        self.time = time_taken

        rhofit_dir = output_dir / "rhofit"
        rhofit_results_path = rhofit_dir / "results.txt"
        rhofit_model_path = rhofit_dir / "best.pdb"

        if ligand_fit_pdb_path.exists():
            self.success = True
            self.result_model_path = [rhofit_model_path]
        else:
            self.success = False
            self.result_model_path = []

        self.rscc = self.parse_rhofit_log(rhofit_results_path)

    def to_json(self):
        record = {}
        record["time"] = self.time
        record["success"] = self.success
        record["result_model_path"] = [str(result_path) for result_path in self.result_model_path]

        json_string = json.dumps(record)
        with open(str(self.output_dir / "task_results.json"), "w") as f:
            f.write(json_string)

    def parse_rhofit_log(self, rhofit_results_path):

        regex = "(Hit_[^\s]+)[\s]+[^\s]+[\s]+[^\s]+[\s]+([^\s]+)"

        with open(str(rhofit_results_path), "r") as f:
            results_string = f.read()

        rscc_matches = re.findall(regex,
                                  results_string,
                                  )

        rsccs = {str(match_tuple[0]): float(match_tuple[1]) for match_tuple in rscc_matches}

        max_rscc = max(list(rsccs.values()))

        return max_rscc


class AutobuildPhenixControlTask(luigi.Task):
    submit_script_path = luigi.Parameter()
    out_dir_path = luigi.Parameter()
    mtz = luigi.Parameter()
    ligand = luigi.Parameter()
    receptor = luigi.Parameter()

    def run(self):
        submit_script_path = Path(self.submit_script_path)
        output_dir = Path(self.out_dir_path)
        target_path = output_dir / "task_results.json"

        command = self.command(self.out_dir_path,
                               self.mtz,
                               self.ligand,
                               self.receptor,
                               )

        start_time = time.time()

        QSub(command,
             submit_script_path,
             )()

        finish_time = time.time()

        status = AutobuildStatus(output_dir,
                                 finish_time - start_time,
                                 )
        status.to_json()

    def command(self,
                out_dir_path,
                mtz,
                ligand,
                receptor,
                ):
        env = "module load phenix"
        ligand_fit_command = "phenix.ligandfit"
        ligand_fit_args = "data={mtz} ligand={ligand} model={receptor}"
        ligand_fit_args_formatted = ligand_fit_args.format(mtz=mtz,
                                                           ligand=ligand,
                                                           receptor=receptor,
                                                           )
        command = "{env}; cd {out_dir_path}; {ligand_fit_command} {args}".format(env=env,
                                                                                 out_dir_path=out_dir_path,
                                                                                 ligand_fit_command=ligand_fit_command,
                                                                                 args=ligand_fit_args_formatted,
                                                                                 )

        return command

    def output(self):
        pandda_done_path = Path(self.out_dir_path) / "task_results.json"
        return luigi.LocalTarget(str(pandda_done_path))


class AutobuildPhenixEventTask(luigi.Task):
    submit_script_path = luigi.Parameter()
    out_dir_path = luigi.Parameter()
    mtz = luigi.Parameter()
    ligand = luigi.Parameter()
    receptor = luigi.Parameter()
    coord = luigi.Parameter()

    def run(self):
        submit_script_path = Path(self.submit_script_path)
        output_dir = Path(self.out_dir_path)
        target_path = output_dir / "task_results.json"

        command = self.command(self.out_dir_path,
                               self.mtz,
                               self.ligand,
                               self.receptor,
                               self.coord,
                               )

        start_time = time.time()

        QSub(command,
             submit_script_path,
             )()

        finish_time = time.time()

        status = AutobuildStatus(output_dir,
                                 finish_time - start_time)
        status.to_json()

    def command(self,
                out_dir_path,
                mtz,
                ligand,
                receptor,
                coord,
                ):
        env = "module load phenix"
        ligand_fit_command = "phenix.ligandfit"
        ligand_fit_args = "data={mtz} ligand={ligand} model={receptor} search_center=[{x},{y},{z}] search_dist=6"
        ligand_fit_args_formatted = ligand_fit_args.format(mtz=mtz,
                                                           ligand=ligand,
                                                           receptor=receptor,
                                                           x=coord[0],
                                                           y=coord[1],
                                                           z=coord[2],
                                                           )
        command = "{env}; cd {out_dir_path}; {ligand_fit_command} {args}".format(env=env,
                                                                                 out_dir_path=out_dir_path,
                                                                                 ligand_fit_command=ligand_fit_command,
                                                                                 args=ligand_fit_args_formatted,
                                                                                 )

        return command

    def output(self):
        pandda_done_path = Path(self.out_dir_path) / "task_results.json"
        return luigi.LocalTarget(str(pandda_done_path))


def process_luigi(tasks,
                  ):
    luigi.build(tasks,
                workers=100,
                local_scheduler=True,
                )


def try_remove(path):
    try:
        shutil.rmtree(path,
                      ignore_errors=True)
    except Exception as e:
        print(e)


def try_make(path):
    try:
        os.mkdir(str(path))
    except Exception as e:
        print(e)


def is_done(original_pandda_output):
    if original_pandda_output.is_dir():
        if (original_pandda_output / "luigi.finished").is_file():
            return True

    return False


def get_ligand_model_path(event: Event):
    # ligand_smiles_path = Path(event.ligand_smiles_path)

    event_dir: Path = Path(event.initial_model_path).parent

    ligands = list((event_dir / "ligand_files").glob("*.pdb"))
    ligand_strings = [str(ligand_path) for ligand_path in ligands if ligand_path.name != "tmp.pdb"]

    ligand_pdb_path: Path = Path(min(ligand_strings,
                                     key=len,
                                     )
                                 )
    return ligand_pdb_path


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

    return output_path


def get_autobuilding_task(event, output_dir):
    try:
        # Directory setup
        autobuilding_dir = output_dir / "{}_{}_{}".format(event.pandda_name,
                                                          event.dtag,
                                                          event.event_idx,
                                                          )
        # try_remove(autobuilding_dir)
        try_make(autobuilding_dir)

        # Prerequisite files
        protein_model_path: Path = event.initial_model_path
        ligand_model_path: Path = get_ligand_model_path(event)
        event_map_path: Path = event.event_map_path
        # resolution: float = event["analysed_resolution"]
        mtz_path = event.data_path
        event_coord = np.array([event.x, event.y, event.z])

        # print("\tPlacing ligand...")
        placed_ligand_path = autobuilding_dir / "ligand.pdb"
        if not (autobuilding_dir / "ligand.pdb").is_file():
            placed_ligand_path = coarse_build(ligand_model_path=ligand_model_path,
                                              event=event,
                                              output_path=placed_ligand_path,
                                              )

        # print("\tStripping receptor waters...")
        stripped_receptor_path = autobuilding_dir / "stripped_receptor.pdb"
        if not stripped_receptor_path.is_file():
            stripped_receptor_path = strip_receptor_waters(receptor_path=protein_model_path,
                                                           placed_ligand_path=placed_ligand_path,
                                                           output_path=stripped_receptor_path,
                                                           )

        # print("\tConverting event map to mtz...")
        event_map_mtz_path = autobuilding_dir / "{}_{}.mtz".format(event.dtag, event.event_idx)
        if not event_map_mtz_path.is_file():
            event_map_mtz_path: Path = event_map_to_mtz(event_map_path,
                                                        event_map_mtz_path,
                                                        event.analysed_resolution,
                                                        )

        # Phenix control
        autobuild_phenix_control_dir = autobuilding_dir / "phenix_control"
        try_make(autobuild_phenix_control_dir)
        autobuild_phenix_control_task = AutobuildPhenixControlTask(
            submit_script_path=autobuild_phenix_control_dir / "submit_phenix_control_autobuild.sh",
            out_dir_path=autobuild_phenix_control_dir,
            mtz=mtz_path,
            ligand=placed_ligand_path,
            receptor=stripped_receptor_path,
        )

        # Phenix autobuild
        autobuild_phenix_event_dir = autobuilding_dir / "phenix_event"
        try_make(autobuild_phenix_event_dir)
        autobuild_phenix_event_task = AutobuildPhenixEventTask(
            submit_script_path=autobuild_phenix_event_dir / "submit_phenix_event_autobuild.sh",
            out_dir_path=autobuild_phenix_event_dir,
            mtz=event_map_mtz_path,
            ligand=placed_ligand_path,
            receptor=stripped_receptor_path,
            coord=event_coord,
        )
        return [autobuild_phenix_control_task, autobuild_phenix_event_task]

    except Exception as e:
        print(e)

        return None


def get_autobuild_tasks(events,
                        output_dir,
                        ):
    task_pairs = joblib.Parallel(n_jobs=25,
                                 verbose=15,
                                 )(joblib.delayed(get_autobuilding_task)(event, output_dir)
                                   for event
                                   in events
                                   )

    tasks = []
    for task_pair in task_pairs:
        if task_pair is not None:
            tasks.append(task_pair[0])
            tasks.append(task_pair[1])

    return tasks


def get_event_table(path):
    events = []
    event_table = pd.read_csv(str(path))
    for idx, event_row in event_table.iterrows():
        if event_row["actually_built"] is True:
            event = Event.from_record(event_row)
            events.append(event)
        else:
            continue

    return events


class RhofitNormal(luigi.Task):
    out_dir_path = luigi.Parameter()
    mtz_path = luigi.Parameter()
    ligand_path = luigi.Parameter()
    model_path = luigi.Parameter()

    def run(self):
        submit_script_path = Path(self.submit_script_path)
        output_dir = Path(self.out_dir_path)
        target_path = output_dir / "task_results.json"

        command = self.command(self.out_dir_path,
                               self.mtz_path,
                               self.ligand,
                               self.ligand_path,
                               )

        start_time = time.time()

        QSub(command,
             submit_script_path,
             )()

        finish_time = time.time()

        status = AutobuildStatusRhofit(output_dir,
                                       finish_time - start_time,
                                       )
        status.to_json()

    def command(self,
                out_dir_path,
                mtz,
                ligand,
                receptor,
                ):
        env = "module load buster"
        ligand_fit_command = "rhofit"
        ligand_fit_args = "-m {mtz} -l {ligand} -p {receptor}"
        ligand_fit_args_formatted = ligand_fit_args.format(mtz=mtz,
                                                           ligand=ligand,
                                                           receptor=receptor,
                                                           )
        command = "{env}; cd {out_dir_path}; {ligand_fit_command} {args}".format(env=env,
                                                                                 out_dir_path=out_dir_path,
                                                                                 ligand_fit_command=ligand_fit_command,
                                                                                 args=ligand_fit_args_formatted,
                                                                                 )

        return command

    def output(self):
        pandda_done_path = Path(self.out_dir_path) / "task_results.json"
        return luigi.LocalTarget(str(pandda_done_path))


def get_rhofit_normal_tasks(events, output_dir):
    tasks = []
    for event in events:
        autobuilding_dir = output_dir / "{}_{}_{}".format(event.pandda_name,
                                                          event.dtag,
                                                          event.event_idx,
                                                          )
        rohfit_normal_dir = autobuilding_dir / "rhofit_normal"

        try_make(rohfit_normal_dir)

        task = RhofitNormal(out_dir=rohfit_normal_dir,
                            mtz_path=event.data_path,
                            model_path=event.initial_model_path,
                            ligand_path=event.ligand_smiles_path,
                            )
        tasks.append(task)

    return tasks


def elbow(autobuilding_dir, ligand_smiles_path):
    command = "module load phenix; cd {autobuilding_dir}; phenix.elbow {ligand_smiles_path}"
    formatted_command = command.format(autobuilding_dir=autobuilding_dir,
                                       ligand_smiles_path=ligand_smiles_path,
                                       )

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    return stdout, stderr


def prepare_event(event, output_dir):
    autobuilding_dir = output_dir / "{}_{}_{}".format(event.pandda_name,
                                                      event.dtag,
                                                      event.event_idx,
                                                      )

    try_make(autobuilding_dir)

    # Get ligand restraints
    ligand_smiles_path = event.ligand_smiles_path
    ligand_cif_path = autobuilding_dir / "ligand.cif"
    if not ligand_cif_path.exists():
        elbow(ligand_smiles_path,
              ligand_cif_path,
              )

    # Convert event map to mtz
    event_map_path = Path(event.event_map_path)
    event_map_mtz_path = autobuilding_dir / "event.mtz"
    if not event_map_path.exists():
        event_map_to_mtz(event_map_path,
                         event_map_mtz_path,
                         event.analysed_resolution,
                         )


def prepare_data(events, output_dir):
    joblib.Parallel(n_jobs=20,
                    verbose=10,
                    )(joblib.delayed(prepare_event)(event,
                                                    output_dir,
                                                    )
                      for event
                      in events
                      )


if __name__ == "__main__":
    print("Parsing args")
    args = parse_args()

    print("Geting Config...")
    config = get_config(args)

    print("Setiting up output...")
    output: Output = setup_output_directory(config.out_dir_path)

    print("Getting event table...")
    events = get_event_table(config.event_table_path)
    print("\tGot {} events!".format(len(events)))

    # Prepare data
    prepare_data(events, config.out_dir_path)

    # Buster normal
    print("Autobuilding events from normal maps with rhofit")
    tasks = get_rhofit_normal_tasks(events,
                                    config.out_dir_path,
                                    )
    tasks[0].run()
    exit()
    process_luigi(tasks)

    # Buster event
    print("Autobuilding events from event maps with rhofit")

    # Phenix normal
    print("Autobuilding events from normal maps with phenix")

    # Phenix Event
    print("Autobuilding events from event maps with phenix")
