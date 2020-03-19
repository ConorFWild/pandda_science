from typing import NamedTuple
import os
import subprocess
import shutil
import argparse
from pathlib import Path

import pandas as pd

from dask.distributed import Client
from dask_jobqueue import SGECluster


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--model_dirs_table_path",
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
    model_dirs_table_path: Path


def get_config(args):
    config = Config(out_dir_path=Path(args.out_dir_path),
                    model_dirs_table_path=Path(args.model_dirs_table_path),
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


def get_model_dirs(path: Path):
    model_dirs_df = pd.read_csv(str(path))
    model_dirs_series = model_dirs_df[model_dirs_df["num_models"]  != 0]["model_dir"]
    model_dirs = [Path(path) for path in model_dirs_series]
    return model_dirs


class DispatchOriginalPanDDA:
    def __init__(self, model_dir, output_dir):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.type = "original"

    def __call__(self):
        command = "module load ccp4; pandda.analyse data_dirs='{data_dir}/*' pdb_style='dimple.pdb' mtz_style='dimple.mtz' cpus=12 out_dir={out_dir}"
        formatted_command = command.format(data_dir=self.model_dir,
                                           out_dir=self.output_dir,
                                           )
        print(formatted_command)
        proc = subprocess.Popen(formatted_command,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                )
        stdout, stderr = proc.communicate()
        print(stdout)
        print(stderr)


class DispatchParallelPanDDA:
    def __init__(self, model_dir, output_dir):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.type = "parallel"

    def __call__(self):
        command = "/dls/science/groups/i04-1/conor_dev/ccp4/build/bin/cctbx.python /dls/science/groups/i04-1/conor_dev/pandda_2/program/run_pandda_2.py data_dirs='{data_dir}/*' pdb_style='dimple.pdb' mtz_style='dimple.mtz' cpus=12 out_dir={out_dir} h_vmem=100 m_mem_free=5 process_dict_n_cpus=12"
        formatted_command = command.format(data_dir=self.model_dir,
                                           out_dir=self.output_dir,
                                           )
        print(formatted_command)
        proc = subprocess.Popen(formatted_command,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                )
        stdout, stderr = proc.communicate()
        print(stdout)
        print(stderr)


def parse_pandda(pandda_output_path: Path):
    pandda_events_path = pandda_output_path / "analyses" / "pandda_analyse_events.csv"
    pandda_events_table = pd.read_csv(str(pandda_events_path))
    record = {}
    record["event_table_path"] = pandda_events_path
    record["num_events"] = len(pandda_events_table)
    record["mean_event_size"] = pandda_events_table["cluster_size"].mean()
    return record


def call(func):
    return func()


def process_dask(funcs,
                 jobs=10,
                 cores=3,
                 processes=3,
                 h_vmem=20,
                 m_mem_free=5,
                 h_rt=3000,
                 ):
    cluster = SGECluster(n_workers=0,
                         job_cls=None,
                         loop=None,
                         security=None,
                         silence_logs='error',
                         name=None,
                         asynchronous=False,
                         interface=None,
                         host=None,
                         protocol='tcp://',
                         dashboard_address=':8787',
                         config_name=None,
                         processes=processes,
                         queue='low.q',
                         project="labxchem",
                         cores=cores,
                         memory="{}GB".format(h_vmem),
                         walltime=h_rt,
                         resource_spec="m_mem_free={}G,h_vmem={}G,h_rt={}".format(m_mem_free, h_vmem, h_rt),
                         )
    cluster.scale(jobs=jobs)
    client = Client(cluster)
    results_futures = client.map(call,
                                 funcs,
                                 )
    results = client.gather(results_futures)

    return results


def pandda_fail(parallel_pandda_table,
                model_dir,
                type):
    parallel_pandda_table_model_dir = parallel_pandda_table[parallel_pandda_table["model_dir"] == model_dir]
    parallel_pandda_table_model_dir_type = parallel_pandda_table_model_dir[
        parallel_pandda_table_model_dir["type"] == type]
    if len(parallel_pandda_table_model_dir_type) == 1:
        print(parallel_pandda_table_model_dir_type)
        if parallel_pandda_table_model_dir_type["failed"] == 1:
            return True

    return False


def try_remove(path):
    try:
        shutil.rmtree(path,
                      ignore_errors=True)
    except Exception as e:
        print(e)


def get_pandda_tasks(model_dirs,
                     parallel_pandda_table,
                     ):
    tasks = []
    for model_dir in model_dirs:
        # Original PanDDA
        original_pandda_output = model_dir.parent / "test_pandda_original"
        if pandda_fail(parallel_pandda_table,
                       model_dir,
                       "original"):
            pass
        else:
            try_remove(original_pandda_output)
            original_pandda_tasks = DispatchOriginalPanDDA(model_dir=Path(model_dir),
                                                           output_dir=original_pandda_output,
                                                           )
            tasks.append(original_pandda_tasks)

        # Parallel PanDDA
        parallel_pandda_output = model_dir.parent / "test_pandda_parallel"
        if pandda_fail(parallel_pandda_table,
                       model_dir,
                       "parallel"):
            pass
        else:
            try_remove(original_pandda_output)
            parallel_pandda_tasks = DispatchParallelPanDDA(model_dir=model_dir,
                                                           output_dir=parallel_pandda_output,
                                                           )
            tasks.append(parallel_pandda_tasks)

    return tasks


def get_status_df(tasks):
    records = []
    for task in tasks:
        record = {}
        record["model_dir"] = task.model_dir
        record["type"] = task.type
        record["output_dir"] = task.output_dir

        # CHeck if ran
        if (task.output_dir / "analyses").is_dir():
            record["ran"] = 1
        else:
            record["ran"] = 0
        # Check if failed
        if (task.output_dir / "pandda.errored").is_file():
            record["failed"] = 1
        else:
            record["failed"] = 0
        # Check if suceeded
        if (task.output_dir / "analyses" / "pandda_analyse_events.csv").is_file():
            record["suceeded"] = 1
        else:
            record["suceeded"] = 0

        records.append(record)

    status_df = pd.DataFrame(records)

    return status_df


def update_parallel_pandda_table(parallel_pandda_table,
                                 status_df,
                                 ):
    for idx, row in status_df.iterrows():
        # if row in table, skip
        parallel_pandda_table_model_dir = parallel_pandda_table[parallel_pandda_table["model_dir"] == row["model_dir"]]
        parallel_pandda_table_modeland_type_dir = parallel_pandda_table_model_dir[
            parallel_pandda_table_model_dir["type"] == row["type"]]
        if len(parallel_pandda_table_modeland_type_dir) != 0:
            continue

        # If failed, enter record
        if row["failed"] == 1:
            record = {"model_dir": row["model_dir"],
                      "output_dir": row["output_dir"],
                      "suceeded": row["suceeded"],
                      "failed": row["failed"],
                      }
            parallel_pandda_table = parallel_pandda_table.append(pd.DataFrame([record]),
                                                                 ignore_index=True)
            continue
        # If suceeded, enter record
        if row["suceeded"] == 1:
            record = {"model_dir": row["model_dir"],
                      "output_dir": row["output_dir"],
                      "suceeded": row["suceeded"],
                      "failed": row["failed"],
                      }
            record.update(parse_pandda(row["output_dir"]))
            parallel_pandda_table = parallel_pandda_table.append(pd.DataFrame([record]),
                                                                 ignore_index=True,
                                                                 )

    return parallel_pandda_table


if __name__ == "__main__":
    print("Parsing args")
    args = parse_args()

    print("Geting Config...")
    config = get_config(args)

    print("Setiting up output...")
    output: Output = setup_output_directory(config.out_dir_path)

    print("Getting model dirs...")
    model_dirs = get_model_dirs(config.model_dirs_table_path)

    print("Checking for old parallel pandda table...")
    if output.parallel_pandda_table_path.is_file():
        parallel_pandda_table: pd.DataFrame = pd.read_csv(str(output.parallel_pandda_table_path))
    else:
        columns = ["model_dir", "type" ]
        parallel_pandda_table: pd.DataFrame = pd.DataFrame(columns=columns)

    print("Making tasks...")
    tasks = get_pandda_tasks(model_dirs,
                             parallel_pandda_table,
                             )

    # Algorithm:
    # until all the model dirs have been done,
    #   spin up x processes,
    #   run a pandda locally on each,
    #   if dask fails,
    #       check if any are pandda fails
    #       regeneate tasks
    #       update and output parallel_pandda_table

    print("Running tasks...")
    while len(parallel_pandda_table) < 2 * len(model_dirs):
        try:
            results = process_dask(tasks,
                                   jobs=8,
                                   cores=12,
                                   processes=1,
                                   h_vmem=120,
                                   m_mem_free=6,
                                   h_rt=3600 * 40,
                                   )

        except Exception as e:
            print(e)
            status_df = get_status_df(tasks)
            parallel_pandda_table = update_parallel_pandda_table(parallel_pandda_table,
                                                                 status_df,
                                                                 )
            tasks = get_pandda_tasks(model_dirs,
                                     parallel_pandda_table,
                                     )
