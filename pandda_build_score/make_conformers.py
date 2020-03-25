from typing import Dict, Tuple, NamedTuple
import os
import shutil

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from functools import partial

from rdkit import Chem
from rdkit.Chem import AllChem


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--event_table_path",
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
    event_table_path: Path
    out_dir_path: Path


def get_training_config(args):
    config = Config(event_table_path=Path(args.event_table_path),
                    out_dir_path=Path(args.out_dir_path),
                    )

    return config


class Output:
    def __init__(self, path: Path):
        self.output_dir = path
        self.output_build_score_training_df_path = path / "output_build_score_training_df.csv"
        self.output_build_score_training_df_path = path / "output_build_score_test_df.csv"

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


def get_event_table(path: Path):
    return pd.read_csv(str(path))


def output_conformer(conformer,
                     output_path,
                     ):
    writer = Chem.PDBWriter(str(output_path))
    writer.write(conformer)


def smiles_from_path(path):
    with open(str(path), "r") as f:
        string = f.read()

    return string


def make_conformers(smiles_path,
                    output_dir,
                    num_confs=20,
                    rms_thresh=2,
                    ):
    smiles_string = smiles_from_path(smiles_path)
    m = Chem.MolFromSmiles(smiles_string)
    m2 = Chem.AddHs(m)
    cids = AllChem.EmbedMultipleConfs(m2,
                                      numConfs=num_confs,
                                      pruneRmsThresh=rms_thresh,
                                      )
    for i, conformer in enumerate(cids):
        output_conformer(conformer,
                         output_dir / "{}.pdb".format(i),
                         )

    return len(cids), output_dir


def process(funcs: Dict):
    results = {}
    for key, func in funcs.items():
        results[key] = func()

    return results


def trymake(conformer_output_dir):
    try:
        os.mkdir(str(conformer_output_dir))
    except Exception as e:
        print(e)


def make_conformer_tasks(event_table,
                         output_dir,
                         ):
    tasks = {}
    for idx, row in event_table.iterrows():
        pandda_name = row["pandda_name"]
        dtag = row["dtag"]
        event_idx = row["event_idx"]
        conformer_output_dir = output_dir / "{}_{}_{}".format(pandda_name,
                                                              dtag,
                                                              event_idx,
                                                              )
        trymake(conformer_output_dir)
        if row["ligand_smiles_path"] == "None":
            continue
        task = partial(make_conformers,
                       row["ligand_smiles_path"],
                       conformer_output_dir,
                       )
        tasks[(pandda_name, dtag, event_idx)] = task

    return tasks


def make_table(results):
    records = []
    for key, result in results.items():
        record = {}
        record["pandda_name"] = key[0]
        record["dtag"] = key[1]
        record["event_idx"] = key[2]
        record["num_confs"] = result[0]
        record["output_dir"] = result[1]
        records.append(record)

    return pd.DataFrame(records)


def output_table(results_table,
                 output_dir):
    results_table.to_csv(str(output_dir))


if __name__ == "__main__":
    args = parse_args()

    config: Config = get_training_config(args)

    output: Output = setup_output(config.out_dir_path)

    event_table = get_event_table(config.event_table_path)

    conformer_tasks = make_conformer_tasks(event_table,
                                           output.output_dir,
                                           )

    results = process(conformer_tasks)

    results_table = make_table(results)
    output_table(results_table,
                 config.out_dir_path / "conformer_table.csv",
                 )
