from __future__ import annotations

from typing import *

from dataclasses import dataclass

import os
import argparse
import re
import subprocess
import shutil
import json
import time
from pathlib import Path
from pprint import PrettyPrinter

import joblib


@dataclass()
class MapperJoblib:
    parallel: Any

    @staticmethod
    def from_joblib(n_jobs=7, verbose=11):
        parallel_env = joblib.Parallel(n_jobs=n_jobs, verbose=verbose).__enter__()
        return MapperJoblib(parallel_env)

    def map_to_list(self, func, *args):
        results = self.parallel(joblib.delayed(func)(*[arg[i] for arg in args])
                                for i, arg
                                in enumerate(args[0])
                                )

        return results

    def map_dict(self, func, *args):
        keys = list(args[0].keys())

        results = self.parallel(joblib.delayed(func)(*[arg[key] for arg in args])
                                for key
                                in keys
                                )

        results_dict = {keys[i]: results[i]
                        for i, key
                        in enumerate(keys)
                        }

        return results_dict


@dataclass()
class MapperPython:
    parallel: Any

    @staticmethod
    def from_python():
        parallel_env = map
        return MapperPython(parallel_env)

    def map_to_list(self, func, *args):
        results = list(self.parallel(func,
                                     *args
                                     ))

        return results


@dataclass()
class QSub:
    qsub_command: str

    # def __init__(self,
    #              command,
    #              submit_script_path,
    #              queue="low.q",
    #              cores=1,
    #              m_mem_free=10,
    #              h_vmem=20,
    #              ):
    #     self.command = command
    #     self.submit_script_path = submit_script_path
    #     self.queue = queue
    #     self.cores = cores
    #     self.m_mem_free = m_mem_free
    #     self.h_vmem = h_vmem
    #
    #     with open(str(submit_script_path), "w") as f:
    #         f.write(command)
    #
    #     chmod_proc = subprocess.Popen("chmod 777 {}".format(submit_script_path),
    #                                   shell=True,
    #                                   stdout=subprocess.PIPE,
    #                                   stderr=subprocess.PIPE,
    #                                   )
    #     chmod_proc.communicate()
    #
    #     qsub_command = "qsub -q {queue} -pe smp {cores} -l m_mem_free={m_mem_free}G,h_vmem={h_vmem}G {submit_script_path}"
    #     self.qsub_command = qsub_command.format(queue=self.queue,
    #                                             cores=self.cores,
    #                                             m_mem_free=self.m_mem_free,
    #                                             h_vmem=self.h_vmem,
    #                                             submit_script_path=self.submit_script_path,
    #                                             )

    def __call__(self):
        # print("\tCommand is: {}".format(self.command))
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

    @staticmethod
    def from_command(command,
                     submit_script_path,
                     queue="low.q",
                     cores=1,
                     m_mem_free=10,
                     h_vmem=20, ):
        with open(str(submit_script_path), "w") as f:
            f.write(command)

        chmod_proc = subprocess.Popen("chmod 777 {}".format(submit_script_path),
                                      shell=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      )
        chmod_proc.communicate()

        qsub_command = "qsub -q {queue} -pe smp {cores} -l m_mem_free={m_mem_free}G,h_vmem={h_vmem}G {submit_script_path}"
        qsub_command = qsub_command.format(queue=queue,
                                           cores=cores,
                                           m_mem_free=m_mem_free,
                                           h_vmem=h_vmem,
                                           submit_script_path=submit_script_path,
                                           )

        return QSub(qsub_command=qsub_command)
