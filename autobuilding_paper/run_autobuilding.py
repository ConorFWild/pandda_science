import argparse
import json
from pprint import PrettyPrinter

from pathlib import Path

import pandas as pd

from pandda_types import logs
from pandda_types.process import QSub

from autobuilding_paper.results import SystemTable, PanDDAResult
from autobuilding_paper.constants import *


class Config:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-i", "--system_file",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        parser.add_argument("-p", "--panddas_file",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        parser.add_argument("-a", "--autobuild_file",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        args = parser.parse_args()

        self.system_file = Path(args.system_file)
        self.panddas_file = Path(args.panddas_file)
        self.autobuild_file = Path(args.autobuild_file)


class Autobuild:
    def __init__(self, model_dir, out_dir, process):
        self.model_dir = model_dir
        self.out_dir = out_dir
        self.process = process

    def poll(self):
        self.process()
        results = self.get_results()
        return results

    def get_results(self):
        finished = self.is_finished()

        events = self.get_events()

        return {"model_dir": self.model_dir,
                "out_dir": self.out_dir,
                "finished": finished,
                "events": events,
                }

    def get_events(self):
        event_table_file = self.out_dir / PANDDA_ANALYSES_DIR / PANDDA_ANALYSE_EVENTS_FILE
        event_table = pd.read_csv(str(event_table_file))

        dictionary = {}
        for index, row in event_table.iterrows():
            series = row.to_dict()
            dictionary[(series["dtag"], series["event_idx"])] = series

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

        command = "{env}; {python} {program} -i {input_pandda} -o {overwrite} -p {version}"

        formatted_command = command.format()

        process = QSub(command,
                       script_path,
                       cores=cpus,
                       m_mem_free=m_mem_free,
                       h_vmem=h_vmem,
                       )

        return PanDDA(model_dir, out_dir, process)


def to_json(dictionary, path):
    with open(str(path), "w") as f:
        json.dump(f,
                  dictionary,
                  )


def main():
    printer = PrettyPrinter()
    config = Config()

    system_table = SystemTable.from_json(config.system_file)

    pandda_table = PanDDAResult.from_json(config.pandda_file)

    autobuilds = {}
    for system_id, system_info in pandda_table.to_dict().items():
        logs.LOG[system_id] = {}
        logs.LOG[system_id]["started"] = True
        printer.pprint(logs.LOG.dict)

        autobuild = Autobuild.from_system(Path(system_info),
                                          config.panddas_dir / system_id,
                                          script_path=Path("/tmp") / system_id,
                                          )
        result = autobuild.poll()

        autobuilds[system_id]["result"] = result

        printer.pprint(logs.LOG.dict)

    to_json(autobuilds,
            config.panddas_dir / "results.json",
            )


if __name__ == "__main__":
    main()
