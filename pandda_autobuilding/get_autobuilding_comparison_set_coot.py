import os
import shutil
import random
import argparse
import signal
import subprocess
import datetime
# from pathlib import Path

import numpy as np
import pandas as pd


# from pandda_types.data import Event

class Event:
    def __init__(self,
                 dtag,
                 event_idx,
                 occupancy,
                 analysed_resolution,
                 high_resolution,
                 interesting,
                 ligand_placed,
                 ligand_confidence,
                 viewed,
                 initial_model_path,
                 data_path,
                 model_dir,
                 pandda_dir,
                 pandda_name,
                 final_model_path,
                 event_map_path,
                 actually_built,
                 ligand_smiles_path,
                 x,
                 y,
                 z,
                 distance_to_ligand_model,
                 event_size,
                 ):
        self.dtag = dtag
        self.event_idx = event_idx
        self.occupancy = occupancy
        self.analysed_resolution = analysed_resolution
        self.high_resolution = high_resolution
        self.interesting = interesting
        self.ligand_placed = ligand_placed
        self.ligand_confidence = ligand_confidence
        self.viewed = viewed
        self.initial_model_path = initial_model_path
        self.data_path = data_path
        self.model_dir = model_dir
        self.pandda_dir = pandda_dir
        self.pandda_name = pandda_name
        self.final_model_path = final_model_path
        self.event_map_path = event_map_path
        self.actually_built = actually_built
        self.ligand_smiles_path = ligand_smiles_path
        self.x = x
        self.y = y
        self.z = z
        self.distance_to_ligand_model = distance_to_ligand_model
        self.event_size = event_size

    @staticmethod
    def from_record(row):
        event = Event(dtag=str(row["dtag"]),
                      event_idx=int(row["event_idx"]),
                      occupancy=float(row["occupancy"]),
                      analysed_resolution=float(row["analysed_resolution"]),
                      high_resolution=float(row["high_resolution"]),
                      interesting=row["interesting"],
                      ligand_placed=row["ligand_placed"],
                      ligand_confidence=row["ligand_confidence"],
                      viewed=row["viewed"],
                      initial_model_path=row["initial_model_path"],
                      data_path=row["data_path"],
                      model_dir=row["model_dir"],
                      pandda_dir=row["pandda_dir"],
                      pandda_name=row["pandda_name"],
                      final_model_path=row["final_model_path"],
                      event_map_path=row["event_map_path"],
                      actually_built=bool(row["actually_built"]),
                      ligand_smiles_path=row["ligand_smiles_path"],
                      x=row["x"],
                      y=row["y"],
                      z=row["z"],
                      distance_to_ligand_model=row["distance_to_ligand_model"],
                      event_size=row["event_size"]
                      )
        return event


class Path:
    def __init__(self, path):
        self.path = path

    def __truediv__(self, other):
        return Path(self.path + "/" + other)

    def __div__(self, other):
        return Path(self.path + "/" + str(other))

    def __repr__(self):
        return str(self.path)

    def glob(self):
        return list(os.listdir(self.path))

    def name(self):
        return os.path.basename(self.path)

    def exists(self):
        return os.path.exists(self.path)


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


class Config:
    def __init__(self,
                 event_table_path,
                 rscc_table_path,
                 autobuilds_dir,
                 out_dir_path,
                 name,
                 ):
        self.event_table_path = event_table_path
        self.rscc_table_path = rscc_table_path
        self.autobuilds_dir = autobuilds_dir
        self.out_dir_path = out_dir_path
        self.name = name


def get_training_config():
    # event_table_path = Path(str(raw_input("Please enter event table path: ")))
    event_table_path = Path("/dls/science/groups/i04-1/conor_dev/experiments/pandda_science/event_table.csv")
    # out_dir_path = Path(str(raw_input("Please enter out_dir_path: ")))
    out_dir_path = Path("/dls/science/groups/i04-1/conor_dev/experiments/pandda_science/autobuilding_comparison")
    # autobuilds_dir = Path(str(raw_input("Please enter autobuilds_dir: ")))
    autobuilds_dir = Path("/dls/labxchem/data/2015/lb13379-1/processing/analysis/TMP_autobuilding")
    name = str(raw_input("Please enter your first name: "))
    # rscc_table_path = Path(str(raw_input("Please enter rscc table path: ")))
    rscc_table_path = "/dls/science/groups/i04-1/conor_dev/experiments/pandda_science/rscc.csv"

    config = Config(event_table_path=event_table_path,
                    out_dir_path=out_dir_path,
                    autobuilds_dir=autobuilds_dir,
                    name=name + datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S"),
                    rscc_table_path=rscc_table_path,
                    )

    return config


class Output:
    def __init__(self, path, name):
        self.output_dir = path
        self.output_table_path = path / "{}.csv".format(name)

    def attempt_mkdir(self, path):
        try:
            os.mkdir(str(path))
        except Exception as e:
            print(e)

    def attempt_remove(self, path):
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


def setup_output(path, overwrite=False):
    output = Output(path)
    output.make(overwrite=overwrite)
    return output


def output_table(table, path):
    table.to_csv(str(path))


def get_events(event_talbe_path):
    event_table = pd.read_csv(str(event_talbe_path))

    events = {}
    for event_id, row in event_table.iterrows():
        event = Event.from_record(row)
        events[(event.pandda_name, event.dtag, event.event_idx)] = event

    return events


def get_rsccs(rscc_table_path):
    rscc_table = pd.read_csv(str(rscc_table_path))

    rsccs = {}
    for rscc_id, row in rscc_table.iterrows():
        rscc = float(row["rscc"])
        rsccs[(row["pandda_name"], row["dtag"], int(row["event_idx"]))] = rscc

    return rsccs


def get_autobuilds(autobuilds_dir):
    autobuilds = {}
    for autobuilt_event in autobuilds_dir.glob():
        event_dir = Path(str(autobuilds_dir)) / str(autobuilt_event)
        phenix_build_dir = event_dir / "phenix_event"
        ligandfit_dir = phenix_build_dir / "LigandFit_run_1_"
        ligand_path = ligandfit_dir / "ligand_fit_1.pdb"
        autobuilds[Path(autobuilt_event).name()] = ligand_path

    return autobuilds


def get_response_table():
    column_names = ["pandda_name", "dtag", "event_idx", "rscc", "response"]
    table = pd.DataFrame(columns=column_names)

    return table


def choose_one(indexed):
    length = len(indexed)
    rand = random.randint(0, length)
    return indexed[rand]


def select_event(events, rsccs):
    print("getting event!")
    # high_rscc_event_keys = list(filter(lambda x: rsccs[x] > 0.7,
    #                                    rsccs))
    high_rscc_event_keys = [key for key in rsccs if rsccs[key] > 0.7]
    print(len(high_rscc_event_keys))
    print("High rscc keys: {}".format(high_rscc_event_keys[list(rsccs.keys())[0]]))
    actually_built_high_rscc_event_keys = list(filter(lambda x: events[x].actually_built == True,
                                                      high_rscc_event_keys))
    print(len(actually_built_high_rscc_event_keys))
    print("RSCC keys actually built: {}".format(actually_built_high_rscc_event_keys[0]))

    events_near_models = list(filter(lambda x: events[x].distance_to_ligand_model < 10.0,
                                     actually_built_high_rscc_event_keys,
                                     ))
    print("Events near model: {}".format(events_near_models[0]))

    event_key = choose_one(events_near_models)
    print(list(events.keys())[0])
    print(event_key)
    event = events[event_key]
    rscc = rsccs[event_key]

    return event, rscc


def write_coot_script(event, autobuild_path):
    coot_script_path = "coot.tmp"
    open_event_map = "g = handle_read_ccp4_map({}, 0)".format(event.event_map_path)
    set_contour_level = "set_last_map_contour_level(1)"
    set_displayed = "set_map_displayed(g, 1)"
    open_handbuilt_model = "h = read_pdb({})".format(event.final_model_path)
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


def setup_coot(event, autobuild_path):
    g = handle_read_ccp4_map(str(event.event_map_path), 0)
    set_last_map_contour_level(1)
    set_map_displayed(g, 1)
    print(event.final_model_path)
    print(autobuild_path)
    h = read_pdb(str(event.final_model_path))
    a = read_pdb(str(autobuild_path))

    set_bond_colour_rotation_for_molecule(h, 100.0)
    set_bond_colour_rotation_for_molecule(a, 300.0)

    set_rotation_centre(float(event.x), float(event.y), float(event.z))

    return g, h, a


def make_shell_command(coot_script_path):
    command = "module load ccp4; coot --no-guano --no-state-script --script {}".format(coot_script_path)
    return command


def open_coot(shell_command):
    process = subprocess.Popen(shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               preexec_fn=os.setsid,
                               )
    return process


def open_event(event, autobuild_path):
    coot_script_path = write_coot_script(event, autobuild_path)

    shell_command = make_shell_command(coot_script_path)

    process = open_coot(shell_command)

    return process


def prompt_response():
    raw_response = input(
        "Please enter '0' if the red model is better, '1' if they are similar/are likely to refine to similar models and '2' if the green model is better and '3' if you encountered an error")

    response = int(raw_response)

    if response not in [0, 1, 2, 3]:
        print("Invalid response! Please try again")
        return prompt_response()
    else:
        return response


def close_process(process):
    print(str(process.stdout.read()))
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def update_table(table, event, rscc, reponse):
    record = {}
    record["pandda_name"] = event.pandda_name
    record["dtag"] = event.dtag
    record["event_idx"] = event.event_idx
    record["rscc"] = rscc
    record["response"] = reponse

    table = pd.concat([table, pd.DataFrame([record])],
                      ignore_index=True,
                      )

    return table


def write_table(table, path):
    table.to_csv(str(path))


def clear_coot(xmap, human_model, autobuilt_model):
    close_molecule(human_model)
    close_molecule(autobuilt_model)
    close_molecule(xmap)


def main():
    # args = parse_args()

    config = get_training_config()

    # output: Output = setup_output(config.out_dir_path)

    events = get_events(config.event_table_path)

    rsccs = get_rsccs(config.rscc_table_path)

    table = get_response_table()

    autobuilds = get_autobuilds(config.autobuilds_dir)

    while True:
        print(len(rsccs))
        print(len(events))
        event, rscc = select_event(events, rsccs)
        autobuild_path = autobuilds["{}_{}_{}".format(event.pandda_name, event.dtag, event.event_idx)]
        if not Path(str(event.event_map_path)).exists():
            print("\t{}".format(event.event_map_path))
            continue
        if not Path(str(event.final_model_path)).exists():
            print("\t{}".format(event.final_model_path))
            continue
        if not Path(str(autobuild_path)).exists():
            print("\t{}".format(autobuild_path))
            continue

        # process = open_event(event,
        #                      autobuilds["{}_{}_{}".format(event.pandda_name, event.dtag, event.event_idx)],
        #                      )
        xmap, human_model, autobuilt_model = setup_coot(event,
                                                        autobuild_path,
                                                        )

        response = prompt_response()

        # close_process(process)
        clear_coot(xmap, human_model, autobuilt_model)

        table = update_table(table,
                     event,
                     rscc,
                     response,
                     )

        write_table(table,
                    config.out_dir_path / "{}.csv".format(config.name),
                    )


if __name__ == "__main__":
    main()
