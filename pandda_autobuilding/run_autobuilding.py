from typing import NamedTuple
import os
import time
import subprocess
import shutil
import re
import argparse
from pathlib import Path

import pandas as pd

import luigi


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


def original_pandda_command(data_dirs,
                            out_dir,
                            pdb_style="dimple.pdb",
                            mtz_style="dimple.mtz",
                            cpus=12,
                            ):
    env = "module load ccp4; module load pymol"
    command = "{env}; pandda.analyse data_dirs='{dds}/*' pdb_style={pst} mtz_style={mst} cpus={cpus} out_dir={odr}"
    formatted_command = command.format(env=env,
                                       dds=data_dirs,
                                       pst=pdb_style,
                                       mst=mtz_style,
                                       cpus=cpus,
                                       odr=out_dir,
                                       )
    return formatted_command


class PanDDAStatus:
    def __init__(self, pandda_out_dir: Path):
        self.pandda_out_dir = pandda_out_dir
        is_finished = self.is_finished(self.pandda_out_dir)
        self.finished: bool = is_finished

    def is_finished(self, pandda_out_dir):
        if pandda_out_dir.is_dir():
            if (pandda_out_dir / "analyses").is_dir():
                if (pandda_out_dir / "analyses" / "pandda_analyse_events.csv"):
                    return True

        return False


def mark_finished(path: Path, status: PanDDAStatus, duration):
    with open(str(path), "w") as f:
        if status.finished:
            f.write("done\n")
            f.write("Duration: {}".format(duration))
        else:
            f.write("failed")
            f.write("Duration: {}".format(duration))


class QSub:
    def __init__(self,
                 command,
                 submit_script_path,
                 queue="low.q",
                 cores=12,
                 m_mem_free=5,
                 h_vmem=120,
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


class OriginalPanDDATask(luigi.Task):
    submit_script_path = luigi.Parameter()
    model_dir = luigi.Parameter()
    output_dir = luigi.Parameter()

    def run(self):
        submit_script_path = Path(self.submit_script_path)
        model_dir = Path(self.model_dir)
        output_dir = Path(self.output_dir)
        target_path = output_dir / "luigi.finished"

        command = original_pandda_command(data_dirs=model_dir,
                                          out_dir=output_dir,
                                          )

        start_time = time.time()

        QSub(command,
             submit_script_path,
             )()

        finish_time = time.time()

        status = PanDDAStatus(output_dir)

        mark_finished(target_path,
                      status,
                      finish_time - start_time
                      )

    def output(self):
        pandda_done_path = self.output_dir / "luigi.finished"
        return luigi.LocalTarget(str(pandda_done_path))


def process_luigi(tasks,
                  jobs=10,
                  cores=3,
                  processes=3,
                  h_vmem=20,
                  m_mem_free=5,
                  h_rt=3000,
                  ):
    luigi.build(tasks,
                workers=5,
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


def get_autobuild_tasks(event_table):
    tasks = []
    for model_dir in model_dirs:
        # Original PanDDA
        original_pandda_output = model_dir.parent / "test_pandda_original"
        if not is_done(original_pandda_output):
            try_remove(original_pandda_output)
            try_make(original_pandda_output)
            original_pandda_tasks = OriginalPanDDATask(submit_script_path=original_pandda_output / "sumbit.sh",
                                                       model_dir=Path(model_dir),
                                                       output_dir=original_pandda_output,
                                                       )
            tasks.append(original_pandda_tasks)
        else:
            print("\tAlready done {}".format(original_pandda_output))

        # Parallel PanDDA
        parallel_pandda_output = model_dir.parent / "test_pandda_parallel"
        if not is_done(parallel_pandda_output):

            try_remove(parallel_pandda_output)
            try_make(parallel_pandda_output)
            parallel_pandda_tasks = ParallelPanDDATask(submit_script_path=parallel_pandda_output / "submit.sh",
                                                       model_dir=model_dir,
                                                       output_dir=parallel_pandda_output,
                                                       )
            tasks.append(parallel_pandda_tasks)
        else:
            print("\tAlready done {}".format(parallel_pandda_output))

    return tasks


if __name__ == "__main__":
    print("Parsing args")
    args = parse_args()

    print("Geting Config...")
    config = get_config(args)

    print("Setiting up output...")
    output: Output = setup_output_directory(config.out_dir_path)

    print("Getting event table...")
    event_table = get_event_table()

    print("Making tasks...")
    tasks = get_autobuild_tasks(event_table)
    process_luigi(tasks,
                  jobs=8,
                  cores=12,
                  processes=1,
                  h_vmem=120,
                  m_mem_free=6,
                  h_rt=3600 * 40,
                  )
