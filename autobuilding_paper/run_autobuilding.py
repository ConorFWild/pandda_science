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
    def __init__(self, pandda_dir, process):
        self.pandda_dir = pandda_dir
        self.process = process

    def poll(self):
        if not self.is_finished():
            self.process()
        results = self.get_results()
        return results

    def get_results(self):
        events = self.get_events()

        return {"pandda_dir": self.pandda_dir,
                "finished": True,
                "events": events,
                }

    def is_finished(self):
        if (self.out_dir / PANDDA_ANALYSES_DIR / PANDDA_ANALYSE_EVENTS_FILE).exists():
            return True
        else:
            return False

    @staticmethod
    def from_system(pandda_dir,
                    cpus=12,
                    h_vmem=240,
                    m_mem_free=12,
                    script_path=Path("/tmp"),
                    ):

        command = "{env}; {python} {program} -i {input_pandda} -o {overwrite} -p {version}"

        env="module load gcc/4.9.3; source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_clipper_no_mkl"
        python = "python"
        program = "/dls/science/groups/i04-1/conor_dev/pandda_science"
        input_pandda = pandda_dir
        overwrite = 0
        version = 2

        formatted_command = command.format(env=env,
                                           python=python,
                                           program=program,
                                           input_pandda=input_pandda,
                                           overwrite=overwrite,
                                           version=version,
                                           )

        process = QSub(formatted_command,
                       script_path,
                       cores=cpus,
                       m_mem_free=m_mem_free,
                       h_vmem=h_vmem,
                       )

        return Autobuild(pandda_dir, process)


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
