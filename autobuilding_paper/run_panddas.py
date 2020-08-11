import argparse
import json
from pprint import PrettyPrinter

from pathlib import Path

import pandas as pd

from pandda_types import logs
from pandda_types.process import QSub

from autobuilding_paper.results import SystemTable
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

        parser.add_argument("-p", "--panddas_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        args = parser.parse_args()

        self.system_file = Path(args.system_file)
        self.panddas_dir = Path(args.panddas_dir)


class PanDDA:
    def __init__(self, model_dir, out_dir, process):
        self.model_dir = model_dir
        self.out_dir = out_dir
        self.process = process

    def poll(self):
        if not self.is_finished():
            self.process()
        results = self.get_results()
        return results

    def get_results(self):
        finished = self.is_finished()

        events = self.get_events()

        return {"model_dir": str(self.model_dir),
                "out_dir": str(self.out_dir),
                "finished": finished,
                "events": events,
                }

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
        env = "module load ccp4"
        python = "/dls/science/groups/i04-1/conor_dev/ccp4/build/bin/cctbx.python"
        script = "/dls/science/groups/i04-1/conor_dev/pandda_2/program/run_pandda_2.py"
        pandda_args = "data_dirs='{dds}/*' pdb_style={pst} mtz_style={mst} cpus={cpus} out_dir={odr}".format(
            dds=model_dir,
            pst=pdb_style,
            mst=mtz_style,
            cpus=cpus,
            odr=out_dir,
        )
        qsub_args = "h_vmem={hvm} m_mem_free={mmf} process_dict_n_cpus={cpus}".format(hvm=h_vmem,
                                                                                      mmf=m_mem_free,
                                                                                      cpus=cpus,
                                                                                      )
        command = "{env}; {pyt} {scrp} {pandda_args} {qsub_args}".format(env=env,
                                                                         pyt=python,
                                                                         scrp=script,
                                                                         pandda_args=pandda_args,
                                                                         qsub_args=qsub_args,
                                                                         )

        process = QSub(command,
                       script_path,
                       cores=cpus,
                       m_mem_free=m_mem_free,
                       h_vmem=h_vmem,
                       )

        return PanDDA(model_dir, out_dir, process)


def to_json(dictionary, path):
    print(dictionary)
    with open(str(path), "w") as f:
        json.dump(dictionary,
                  f,
                  )


def main():
    printer = PrettyPrinter(depth=2)
    config = Config()

    system_table = SystemTable.from_json(config.system_file)

    panddas = {}
    for system_id, system_info in system_table.to_dict().items():
        if system_info is None:
            logs.LOG[system_id] = {}
            logs.LOG[system_id]["path"] = system_info
            logs.LOG[system_id]["started"] = False
            logs.LOG[system_id]["result"] = None
            continue

        logs.LOG[system_id] = {}
        logs.LOG[system_id]["path"] = system_info
        logs.LOG[system_id]["started"] = True
        printer.pprint(logs.LOG.dict)

        pandda = PanDDA.from_system(Path(system_info),
                                    config.panddas_dir / system_id,
                                    script_path=Path("/tmp") / system_id,
                                    )
        result = pandda.poll()

        panddas[system_id] = result
        logs.LOG[system_id]["result"] = result
        printer.pprint(logs.LOG.dict)

    to_json(panddas,
            config.panddas_dir / "results.json",
            )


if __name__ == "__main__":
    main()
