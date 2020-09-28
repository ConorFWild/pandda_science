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
class Dtag:
    dtag: str

    def __hash__(self):
        return hash(self.dtag)

    def to_python(self):
        return self.dtag


@dataclass()
class EventIDX:
    event_idx: int

    def __hash__(self):
        return hash(self.event_idx)

    def to_python(self):
        return self.event_idx


@dataclass()
class EventID:
    dtag: Dtag
    event_idx: EventIDX

    def __hash__(self):
        return hash((self.dtag, self.event_idx))

    def to_python(self):
        return (self.dtag.to_python(), self.event_idx.to_python())


@dataclass()
class Event:
    dtag: Dtag
    event_idx: EventIDX

    def to_python(self):
        return {Event.keys().dtag: self.dtag.to_python(),
                Event.keys().event_idx: self.event_idx.to_python(),
                }

    @staticmethod
    def from_python(py_event):
        dtag: Dtag = Dtag(py_event[Event.keys().dtag])
        event_idx: EventIDX = EventIDX(py_event[Event.keys().event_idx])
        return Event(dtag, event_idx)

    @classmethod
    def keys(cls):
        @dataclass
        class Keys:
            dtag: str = "dtag"
            event_idx: str = "dtag"

        return Keys()


@dataclass()
class PanDDA:
    model_dir: Path
    out_dir: Path
    process: QSub

    def poll(self):
        is_finished = self.is_finished()

        if not is_finished:
            self.process()
        # results = self.get_results()
        is_finished = self.is_finished()

        return is_finished

    # def get_results(self):
    #     finished = self.is_finished()
    #
    #     events = self.get_events()
    #
    #     # return {"model_dir": str(self.model_dir),
    #     #         "out_dir": str(self.out_dir),
    #     #         "finished": finished,
    #     #         "events": events,
    #     #         }
    #
    #     return PanDDAResult(model_dir=self.model_dir,
    #                         out_dir=self.out_dir,
    #                         finished=finished,
    #                         events=events,
    #                         )

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
                    script_path,
                    pdb_style="dimple.pdb",
                    mtz_style="dimple.mtz",
                    cpus=12,
                    h_vmem=240,
                    m_mem_free=12,
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
class Events:
    events: Dict[Dtag, Dict[EventIDX, Event]]

    def __iter__(self):
        for event_id in self.events:
            yield event_id

    def __getitem__(self, item):
        return self.events[item]

    @staticmethod
    def from_pandda_process(pandda_process: PanDDA):
        event_table_file = pandda_process.out_dir / PANDDA_ANALYSES_DIR / PANDDA_ANALYSE_EVENTS_FILE
        event_table = pd.read_csv(str(event_table_file))

        events: Dict[Dtag, Dict[EventIDX, Event]] = {}
        for index, row in event_table.iterrows():
            series = row.to_dict()
            dtag: Dtag = Dtag(series["dtag"])
            event_idx: EventIDX = EventIDX(int(series["event_idx"]))
            event_id: EventID = EventID(dtag, event_idx)

            event: Event = Event(dtag=dtag,
                                 event_idx=event_idx,
                                 )

            if dtag not in events:
                events[dtag] = {}

            events[dtag][event_idx] = event

        return Events(events)

    def to_python(self):
        py_dict = {}
        for dtag in self.events:
            for event_idx in self.events[dtag]:
                py_dtag = dtag.to_python()
                py_event_idx = event_idx.to_python()

                if dtag not in py_dict:
                    py_dict[dtag] = {}

                py_dict[py_dtag][py_event_idx] = self.events[dtag][event_idx].to_python()

        return py_dict

    @classmethod
    def from_python(cls, py_dict):
        events = {}
        for py_dtag in py_dict:
            dtag = Dtag(py_dtag)
            for py_event_idx in py_dict[py_dtag]:
                event_idx = EventIDX(py_event_idx)

                if dtag not in events:
                    events[dtag] = {}
                #
                py_event = py_dict[py_dtag][py_event_idx]
                event = Event.from_python(py_event)

                events[dtag][event_idx] = event

        return Events(events)


@dataclass()
class PanDDAResult:
    model_dir: Path
    out_dir: Path
    finished: bool
    events: Events

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

    def to_python(self):
        dictionary = {
            "model_dir": str(self.model_dir),
            "out_dir": str(self.out_dir),
            "finished": self.finished,
            "events": self.events.to_python(),
        }
        return dictionary

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

    @staticmethod
    def from_pandda(pandda: PanDDA):
        finished: bool = pandda.is_finished()

        events: Events = Events.from_pandda_process(pandda)

        return PanDDAResult(model_dir=pandda.model_dir,
                            out_dir=pandda.out_dir,
                            finished=finished,
                            events=events,
                            )

    @classmethod
    def from_python(cls, py_dict):
        keys = PanDDAResult.keys()
        model_dir = Path(py_dict[keys.model_dir])
        out_dir= Path(py_dict[keys.out_dir])
        finished = bool(py_dict[keys.finished])
        events = Events.from_python(py_dict[keys.events])

        return PanDDAResult(model_dir,
                            out_dir,
                            finished,
                            events,
                            )

    @classmethod
    def keys(cls):
        @dataclass
        class Keys:
            model_dir: str = "model_dir"
            out_dir: str = "out_dir"
            finished: str = "finished"
            events: str = "events"

        return Keys()
