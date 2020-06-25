import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import seaborn

import gemmi

import luigi

from pandda_autobuilding.constants import *
from pandda_types.data import Event
from pandda_types.process import (Rhofit,
                                  AutobuildingResultRhofit,
                                  Elbow,
                                  MapToMTZ,
                                  Strip,
                                  Graft,
                                  QSub,
                                  )


def get_ligand_smiles(pandda_event_dir):
    compound_dir = pandda_event_dir / "ligand_files"

    ligand_pdbs = list(compound_dir.glob("*.pdb"))
    ligand_pdb_strings = [str(ligand_path) for ligand_path in ligand_pdbs if ligand_path.name != "tmp.pdb"]
    if len(ligand_pdb_strings) > 0:
        shortest_ligand_path = min(ligand_pdb_strings,
                                   key=len,
                                   )
        return Path(shortest_ligand_path)

    smiles_paths = compound_dir.glob("*.smiles")
    smiles_paths_list = list(smiles_paths)

    if len(smiles_paths_list) > 0:
        return Path(min([str(ligand_path) for ligand_path in smiles_paths_list if ligand_path.name != "tmp.smiles"],
                        key=len)
                    )
    else:
        raise Exception("No smiles found! Smiles list is: {}".format(smiles_paths_list))


def get_events(path):
    events = {}
    event_table = pd.read_csv(str(path))
    for idx, event_row in event_table.iterrows():
        if event_row["actually_built"] is True:
            pandda_processed_dir = Path(event_row["event_map_path"]).parent
            print(pandda_processed_dir)
            if pandda_processed_dir.exists():
                try:
                    event_row["ligand_smiles_path"] = get_ligand_smiles(Path(event_row["event_map_path"]).parent)
                    print("\tLigand smiles path:{}".format(event_row["ligand_smiles_path"]))
                    event = Event.from_record(event_row)
                    # events.append(event)
                    events[(event.pandda_name, event.dtag, event.event_idx)] = event

                except Exception as e:
                    print(e)
                    continue

        else:
            continue

    return events


class Config:

    def __init__(self):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-i", "--event_table_path",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        parser.add_argument("-a", "--autobuilds_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        parser.add_argument("-o", "--out_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        args = parser.parse_args()

        self.event_table_path = args.event_table_path
        self.autobuilds_dir = Path(args.autobuilds_dir)
        self.out_dir = Path(args.out_dir)


def load_available_results_events(events, out_dir_path):
    results = {}
    for event in events.values():
        results_normal_path = Path(out_dir_path) / BUILD_DIR_PATTERN.format(
            pandda_name=event.pandda_name,
            dtag=event.dtag,
            event_idx=event.event_idx,
        ) / RHOFIT_RESULT_JSON_FILE
        if results_normal_path.exists():
            # print("\tLoading result at {}".format(results_normal_path))
            result = AutobuildingResultRhofit.from_json(results_normal_path)
            results[(event.pandda_name, event.dtag, event.event_idx)] = result
        else:
            print("\tNo result at {}".format(results_normal_path))

    return results


def load_available_results_normal(events, out_dir_path):
    results = {}
    for event in events.values():
        results_normal_path = Path(out_dir_path) / BUILD_DIR_PATTERN.format(
            pandda_name=event.pandda_name,
            dtag=event.dtag,
            event_idx=event.event_idx,
        ) / RHOFIT_NORMAL_RESULT_JSON_FILE
        if results_normal_path.exists():
            # print("\tLoading result at {}".format(results_normal_path))
            result = AutobuildingResultRhofit.from_json(results_normal_path)
            results[(event.pandda_name, event.dtag, event.event_idx)] = result
        else:
            print("\tNo result at {}".format(results_normal_path))

    return results


def get_table(results_event, results_normal):
    records = []
    for key in results_event:
        if key in results_normal:
            record = {}
            record["pandda_name"] = key[0]
            record["dtag"] = key[1]
            record["event_idx"] = key[2]
            record["event_rscc"] = results_event[key].rscc
            record["normal_rscc"] = results_normal[key].rscc
            records.append(record)

    return pd.DataFrame(records)


def get_num_better(series1, series2):
    return len(series1[series1 > series2])


def get_delta_modelable(series1, series2, cutoff=0.7):
    series_1_cutoff = series1[series1 > cutoff]
    series_2_cutoff = series2[series1 > cutoff]

    return len(series_2_cutoff[series_2_cutoff < cutoff])


def get_rmsd(ligand_residue_human,
             ligand_residue_autobuilt,
             ):
    distances = []
    for atom_num in range(len(ligand_residue_human)):
        atom_1 = ligand_residue_human[atom_num]
        atom_2 = ligand_residue_autobuilt[atom_num]

        if atom_1.element != atom_2.element:
            raise Exception("Atoms not equivilent!")

        distance = atom_1.pos(atom_2.pos)
        distances.append(distance)

    return np.sqrt(np.sum(np.square(distances)) / len(distances))


def get_lig(model):
    for chain in model:
        for residue in chain:
            if residue.name == "LIG":
                return residue

    raise Exception("No lig!")


def get_rmsds_normal(events,
                     results,
                     out_dir,
                     ):
    rmsds = {}

    for key in results:
        # Load models
        human_model_path = events[key].final_model_path
        print("\tLoading model from {}".format(human_model_path))
        human_model = gemmi.read_structure(human_model_path)

        autobuilt_model_path = str(out_dir / BUILD_DIR_PATTERN.format(
            pandda_name=events[key].pandda_name,
            dtag=events[key].dtag,
            event_idx=events[key].event_idx,
        ) / RHOFIT_NORMAL_DIR / RHOFIT_BEST_MODEL_FILE)
        print("\tLoading autobuilt model path: {}".format(autobuilt_model_path))
        autobuilt_model = gemmi.read_structure(autobuilt_model_path)

        # Get residues
        ligand_residue_human = get_lig(human_model[0])
        ligand_residue_autobuilt = get_lig(autobuilt_model[0])

        # Calculate rmsds
        rmsd = get_rmsd(ligand_residue_human,
                        ligand_residue_autobuilt,
                        )

        # Append
        rmsds[key] = rmsd

    return rmsds


def rscc_scatter(normal_rsccs,
                 event_rsccs,
                 plot_path,
                 ):
    plot = seaborn.scatterplot(normal_rsccs,
                               event_rsccs,
                               )

    fig = plot.get_figure()
    fig.savefig(str(plot_path))


if __name__ == "__main__":
    print("Geting Config...")
    config = Config()

    print("Getting event table...")
    events = get_events(config.event_table_path)
    print("\tGot {} events!".format(len(events)))

    results_event = load_available_results_events(events,
                                                  config.autobuilds_dir,
                                                  )
    print("\t {} event results available".format(len(results_event)))
    results_normal = load_available_results_normal(events,
                                                   config.autobuilds_dir,
                                                   )
    print("\t {} normal results available".format(len(results_normal)))

    table = get_table(results_event, results_normal)
    print(table.head(20))
    delta = (table["event_rscc"] - table["normal_rscc"]).mean()
    print("Detal is {}".format(delta))

    # # Compare RSCCs
    num_event_better = get_num_better(table["event_rscc"], table["normal_rscc"])
    print("Event rscc better for {} out of {} builds".format(num_event_better, len(table)))

    num_normal_better = get_num_better(table["normal_rscc"], table["event_rscc"])
    print("Normal rscc better for {} out of {} builds".format(num_normal_better, len(table)))

    delta_modelable = get_delta_modelable(table["event_rscc"], table["normal_rscc"])
    print("In event {} are modelable that were not".format(delta_modelable))

    delta_modelable = get_delta_modelable(table["normal_rscc"], table["event_rscc"])
    print("In normal {} are modelable that were not".format(delta_modelable))

    cutoff = 0.7
    above_cutoff_normal = len(table["normal_rscc"][table["normal_rscc"] > cutoff])
    print("In normal, {} builds above rscc {}".format(above_cutoff_normal, cutoff))

    above_cutoff_event = len(table["event_rscc"][table["event_rscc"] > cutoff])
    print("In event, {} builds above rscc {}".format(above_cutoff_event, cutoff))

    print("Saving rscc plot")
    rscc_scatter(table["normal_rscc"],
                 table["event_rscc"],
                 config.out_dir / "rscc.png",
                 )

    # # Compare RMSDs
    rmsds_normal = get_rmsds_normal(events,
                                    results_normal,
                                    out_dir=config.autobuilds_dir,
                                    )
    print("Mean normal rmsd to human model is: {}".format(np.mean([rmsd for rmsd in rmsds_normal])))

    # plot_rsccs()
