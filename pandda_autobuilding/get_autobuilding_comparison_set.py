from typing import Dict, Tuple, NamedTuple
import os
import shutil
import random
import argparse
import signal
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import pandda_types


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--event_table",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-r", "--rscc_table",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-a", "--autobuilds_dir",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-o", "--out_dir_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    parser.add_argument("-n", "--name",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    args = parser.parse_args()

    return args


class Config(NamedTuple):
    event_table_path: Path
    rscc_table_path: Path
    autobuilds_dir: Path
    out_dir_path: Path
    name: str


def get_training_config(args):
    config = Config(event_table_path=Path(args.event_table),
                    out_dir_path=Path(args.out_dir_path),
                    autobuilds_dir=args.autobuilds_dir,
                    name=str(args.name),
                    rscc_table_path=Path(args.rscc_table),
                    )

    return config


class Output:
    def __init__(self, path: Path, name: str):
        self.output_dir = path
        self.output_table_path = path / "{}.csv".format(name)

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
            self.attempt_remove(self.output_dir)

        # Make output dirs
        self.attempt_mkdir(self.output_dir)


def setup_output(path: Path, overwrite: bool = False):
    output: Output = Output(path)
    output.make(overwrite=overwrite)
    return output


def output_table(table, path):
    table.to_csv(str(path))


def get_events(event_talbe_path):
    event_table = pd.read_csv(str(event_talbe_path))

    events = {}
    for event_id, row in event_table.iterrows():
        event = pandda_types.Event.from_record(row)
        events[(event.pandda_name, event.dtag, event.event_idx)] = event

    return events


def get_rsccs(rscc_table_path):
    rscc_table = pd.read_csv(str(rscc_table_path))

    rsccs = {}
    for rscc_id, row in rscc_table:
        rscc = float(row["rscc"])
        rsccs[(row["pandda_name"], row["dtag"], int(row["event_idx"]))] = rscc

    return rsccs


def get_autobuilds(autobuilds_dir):
    autobuilds = {}
    for autobuilt_event in autobuilds_dir.glob("*"):
        event_dir = autobuilt_event / "phenix_event"
        ligandfit_dir = event_dir / "LigandFit_run_1_"
        ligand_path = ligandfit_dir / "ligand_fit_1.pdb"
        autobuilds[autobuilt_event.name] = ligand_path

    return autobuilds


def get_response_table():
    column_names = ["pandda_name", "dtag", "event_idx", "rscc", "response"]
    table = pd.DataFrame(columns=column_names)

    return table


def choose_one(indexed):
    length = len(indexed)
    rand = random.randint(length)
    return indexed[rand]


def select_event(events, rsccs):
    high_rscc_event_keys = list(filter(lambda x: x[1] > 0.7, rsccs))

    event_key = choose_one(high_rscc_event_keys)

    event = events[event_key]
    rscc = rsccs[event_key]

    return event, rscc


def write_coot_script(event, autobuild_path):
    coot_script_path = "coot.tmp"
    open_event_map = "g = handle_read_ccp4_map({}, 0)".format(event.event_map_path)
    set_contour_level = "set_last_map_contour_level(1)"
    set_displayed = "set_map_displayed(g, 1)"
    open_handbuilt_model = "h = read_pdb({})".format(event.model_path)
    open_autobuilt_model = "a = read_pdb({})".format(autobuild_path)

    with open(coot_script_path, "w") as f:
        f.write("{}\n{}\n{}\n{}\n{}\n".format(open_event_map,
                                              set_contour_level,
                                              set_displayed,
                                              open_handbuilt_model,
                                              open_autobuilt_model,
                                              )
                )

    return coot_script_path


def make_shell_command(coot_script_path):
    command = "module load ccp4; coot --no-guano --no-state-script --script {}".format(coot_script_path)
    return command


def open_coot(shell_command):
    process = subprocess.Popen(shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def open_event(event, autobuild_path):
    coot_script_path = write_coot_script(event, autobuild_path)

    shell_command = make_shell_command(coot_script_path)

    process = open_coot(shell_command)

    return process


def prompt_response():
    raw_response = input("Please enter '0' if model 1 is better, '1' if they are similar and '2' if model 2 is better")

    response = int(raw_response)

    if response not in [0, 1, 2]:
        print("Invalid response! Please try again")
        return prompt_response()
    else:
        return response


def close_process(process):
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def update_table(table, event, rscc, reponse):
    record = {}
    record["pandda_name"] = event.pandda_name
    record["dtag"] = event.pandda_name
    record["event_idx"] = event.pandda_name
    record["rscc"] = event.pandda_name
    record["response"] = event.pandda_name

    table = table.append(record)

    return table


def write_table(table, path):
    table.to_csv(str(path))


def main():
    args = parse_args()

    config: Config = get_training_config(args)

    # output: Output = setup_output(config.out_dir_path)

    events = get_events(config.event_table_path)

    rsccs = get_rsccs(config.rscc_table_path)

    table = get_response_table()

    autobuilds = get_autobuilds(config.autobuilds_dir)

    while True:
        event, rscc = select_event(events, rsccs)

        process = open_event(event,
                             autobuilds["{}_{}_{}".format(event.pandda_name, event.dtag, event.event_idx)],
                             )

        response = prompt_response()

        close_process(process)

        update_table(table,
                     event,
                     rscc,
                     response,
                     )

        write_table(table,
                    config.out_dir_path / "{}.csv".format(config.name),
                    )


if __name__ == "__main__":
    main()
