from typing import NamedTuple
import os
import shutil
import argparse
from pathlib import Path
import subprocess

import pandas as pd

from dask.distributed import Client
from dask_jobqueue import SGECluster


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-m", "--model_dirs_table_path",
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
    our_dir_path: Path
    model_dirs_table_path: Path


def get_config(args):
    config = Config(our_dir_path=Path(args.out_dir_path),
                    model_dirs_table_path=Path(args.model_dirs_table_path),
                    )

    return config


class Output:
    def __init__(self, out_dir_path: Path):
        self.out_dir_path: Path = out_dir_path
        self.dataset_out_dir_path = out_dir_path / "dataset"
        self.parallel_pandda_out_dir_path = out_dir_path / "parallel_pandda"
        self.dataset_clustering_out_dir_path = out_dir_path / "dataset_clustering"
        self.autobuilding_out_dir_path = out_dir_path / "autobuilding"
        self.build_score_out_dir_path = out_dir_path / "build_score"

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
    model_dirs_series = model_dirs_df[model_dirs_df["num_models"] != 0]["model_dir"]
    model_dirs = [Path(path) for path in model_dirs_series]
    return model_dirs


class DatsetClusteringTask:
    def __init__(self, model_dir, output_dir):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.labelled_csv_path = output_dir / "processing" / "labelled_embeding.csv"

    def __call__(self):
        # Check if already ran
        if self.output_dir.is_dir():
            if self.labelled_csv_path.is_file():
                return None

        try:
            os.mkdir(self.output_dir)
        except Exception as e:
            pass

        env = "module load gcc/4.9.3; source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_clipper_no_mkl"
        python = "python"
        script = "/dls/science/groups/i04-1/conor_dev/dataset_clustering/program/cluster_xcdb_single.py"
        n_processes = 10
        mtz_regex = "dimple.mtz"
        pdb_regex = "dimple.pdb"
        structure_factors = "FWT,PHWT"

        command = "{env}; {python} {script} -r={input} -o={output} -n={n_processes} --mtz_regex={mtz_regex} --pdb_regex={pdb_regex} --structure_factors={structure_factors}"
        formatted_command = command.format(env=env,
                                           python=python,
                                           script=script,
                                           input=self.model_dir,
                                           output=self.output_dir,
                                           n_processes=n_processes,
                                           mtz_regex=mtz_regex,
                                           pdb_regex=pdb_regex,
                                           structure_factors=structure_factors,
                                           )
        print(formatted_command)
        proc = subprocess.Popen(formatted_command,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                )
        stdout, stderr = proc.communicate()

        return None


def get_dataset_clustering_tasks(model_dirs, output_dir):
    tasks = []
    for model_dir_path in model_dirs:
        out_dir = model_dir_path / "pandda_dataset_clustering"
        print("\tSetting up from {} to output in {}".format(model_dir_path, out_dir))
        task = DatsetClusteringTask(model_dir_path,
                                    out_dir,
                                    )
        tasks.append(task)

    return tasks


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
                         job_extra=['-pe smp {}'.format(cores)],
                         )
    cluster.scale(jobs=jobs)
    client = Client(cluster)
    results_futures = client.map(call,
                                 funcs,
                                 )
    results = client.gather(results_futures)

    return results


if __name__ == "__main__":
    args = parse_args()

    config = get_config(args)

    output: Output = setup_output_directory(config.our_dir_path)

    print("Getting model dirs...")
    model_dirs = get_model_dirs(config.model_dirs_table_path)

    print("Making tasks...")
    tasks = get_dataset_clustering_tasks(model_dirs,
                                         output.out_dir_path,
                                         )
    print("\tTasks to process are: {}".format(tasks))

    print("Processing...")
    results = process_dask(tasks,
                           jobs=8,
                           cores=10,
                           processes=1,
                           h_vmem=120,
                           m_mem_free=6,
                           h_rt=3600 * 40,
                           )
