from __future__ import annotations

from typing import *

from dataclasses import dataclass

import os
import argparse
import re
import subprocess
import shutil
import json
from pathlib import Path
from pprint import PrettyPrinter

import pandas as pd

from autobuilding_paper.process_types import QSub

from autobuilding_paper.autobuild_constants import *


@dataclass()
class Config:
    system_file: Path
    pandda_dir: Path
    result_json_file: Path

    @staticmethod
    def from_args_str(args_str):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-i", "--system_file",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        parser.add_argument("-p", "--pandda_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        parser.add_argument("-j", "--result_json_file",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        args = parser.parse_args(args_str)

        return Config(system_file=Path(args.system_file),
                      pandda_dir=Path(args.pandda_dir),
                      result_json_file=Path(args.result_json_file),
                      )


@dataclass()
class PanDDA:
    model_dir: Path
    out_dir: Path
    process: QSub

    def poll(self):
        if not self.is_finished():
            self.process()
        results = self.get_results()
        return results

    def get_results(self):
        finished = self.is_finished()

        events = self.get_events()

        # return {"model_dir": str(self.model_dir),
        #         "out_dir": str(self.out_dir),
        #         "finished": finished,
        #         "events": events,
        #         }

        return PanDDAResult(model_dir=self.model_dir,
                            out_dir=self.out_dir,
                            finished=finished,
                            events=events,
                            )

    def get_events(self):
        event_table_file = self.out_dir / PANDDA_ANALYSES_DIR / PANDDA_ANALYSE_EVENTS_FILE
        event_table = pd.read_csv(str(event_table_file))

        dictionary = {}
        for index, row in event_table.iterrows():
            series = row.to_dict()
            dtag = series["dtag"]
            event_idx = series["event_idx"]
            if dtag not in dictionary:
                dictionary[dtag] = {}

            dictionary[dtag][event_idx] = series

        return dictionary

    def is_finished(self):
        if (self.out_dir / PANDDA_ANALYSES_DIR / PANDDA_ANALYSE_EVENTS_FILE).exists():
            return True
        else:
            return False

    @staticmethod
    def from_system(model_dir,
                    out_dir,
                    pdb_style="dimple.pdb",
                    mtz_style="dimple.mtz",
                    cpus=12,
                    h_vmem=240,
                    m_mem_free=12,
                    script_path=Path("/tmp"),
                    ):
        env = "source /dls/science/groups/i04-1/software/pandda_0.2.12/ccp4/ccp4-7.0/bin/ccp4.setup-sh; module load pymol/1.8.2.0"
        python = "/dls/science/groups/i04-1/conor_dev/ccp4/build/bin/cctbx.python"
        script = "/dls/science/groups/i04-1/conor_dev/pandda_2/program/run_pandda_2.py"
        pandda_args = "data_dirs='{dds}/*' pdb_style={pst} mtz_style={mst} cpus={cpus} out_dir={odr}".format(
            dds=model_dir,
            pst=pdb_style,
            mst=mtz_style,
            cpus=cpus,
            odr=out_dir,
        )

        command = "{env}; pandda.analyse {pandda_args}".format(env=env,
                                                               pandda_args=pandda_args,
                                                               )

        process = QSub.from_command(command,
                                    script_path,
                                    cores=cpus,
                                    m_mem_free=m_mem_free,
                                    h_vmem=h_vmem,
                                    )

        return PanDDA(model_dir,
                      out_dir,
                      process,
                      )


@dataclass()
class PanDDAResult:
    model_dir: Path
    out_dir: Path
    finished: bool
    events: Dict

    def to_dict(self):
        return {"model_dir": str(self.model_dir),
                "out_dir": str(self.out_dir),
                "finished": self.finished,
                "events": self.events,
                }

    def to_json(self, path):
        with open(str(path), "w") as f:
            json.dump(self.to_dict(),
                      f,
                      )

    @staticmethod
    def from_dict(dictionary):
        return PanDDAResult(model_dir=dictionary["model_dir"],
                            out_dir=dictionary["out_dir"],
                            finished=dictionary["finished"],
                            events=dictionary["events"],
                            )

    @staticmethod
    def from_json(file):
        with open(str(file), "r") as f:
            dictionary = json.load(f)
        return PanDDAResult.from_dict(dictionary)
