import re

import numpy as np
import pandas as pd

import gemmi

from fragalysis_api.xcextracter.getdata import GetPdbData
from fragalysis_api.xcextracter.xcextracter import xcextracter

from pandda_autobuilding.constants import *
from pandda_types import logs


class XCEModel:
    pass


class LocalModel:
    pass


def dtag_to_protein_code(table, dtag):
    pass


def protein_code_to_dtag(protein_code):
    regex = "^([0-9a-zA-Z]+[-]+[0-9a-zA-Z]+)"
    matches = re.findall(regex,
                         protein_code,
                         )

    return matches[0]


def xce_id_to_mol(table, xce_id):
    pass


# def get_distance(mol_1, mol_2):
#     pass


def get_distance(model_1, model_2):
    closest_distances = []
    for atom_1 in model_1:
        distances = []
        for atom_2 in model_2:
            distance = atom_1.pos.dist(atom_2.pos)
            distances.append(distance)

        closest_distance = min(distances)
        closest_distances.append(closest_distance)
    return np.mean(closest_distances)


# def get_analogue_mol(autobuild_models,
#                      reference_model,
#                      ):
#     dtag = protein_code_to_dtag(reference_model)
#     model = dtag_to_model(autobuild_models, dtag)
#     return model

def get_model_paths_from_pandda_dir(pandda_dir):
    processed_models_dir = pandda_dir / PANDDA_PROCESSED_DATASETS_DIR

    processed_model_dirs = processed_models_dir.glob("*")

    paths = {}
    for processed_model_dir in processed_model_dirs:
        dtag = processed_model_dir.name
        model_dir = processed_model_dir / PANDDA_MODELLED_STRUCTURES_DIR
        model_path = model_dir / PANDDA_EVENT_MODEL.format(dtag)
        paths[dtag] = model_path

    return paths


def model_path_to_structure(model_path):
    structure = gemmi.read_structure(str(model_path))
    model = structure
    return model


def get_pandda_models(pandda_dir):
    logs.LOG["paths"] = {}
    model_paths = get_model_paths_from_pandda_dir(pandda_dir)

    models = {}
    for model_id, model_path in model_paths.items():
        if model_path.exists():
            models[model_id] = model_path_to_structure(model_path)


            if model_id not in logs.LOG["paths"]:
                logs.LOG["paths"][model_id] = {}
            logs.LOG["paths"][model_id]["local"] = model_path

            if model_id not in logs.LOG.dict:
                logs.LOG[model_id] = {}
            logs.LOG[model_id]["local_path"] = str(model_path)

    return models


def dtag_to_system(dtag):
    regex = "([^-]+)-[^-]+"
    matches = re.findall(regex,
                         dtag,
                         )

    return matches[0]


def pandda_dir_to_project_code(pandda_dir):
    processed_models_dir = pandda_dir / PANDDA_PROCESSED_DATASETS_DIR

    processed_model_dir = next(processed_models_dir.glob("*"))

    example_dtag = processed_model_dir.name
    print("Example dtag is: {}".format(example_dtag))

    project_name = dtag_to_system(example_dtag)

    return project_name


def get_reference_models(project_code):
    print("Project code is: {}".format(project_code))
    summary = xcextracter(project_code)
    print(summary)
    print("Number of records is: {}".format(len(summary)))

    pdb_grabber = GetPdbData()

    reference_models = {}

    for index, row in summary.iterrows():
        protein_code = row["protein_code"]
        # print(protein_code)
        dtag = protein_code_to_dtag(protein_code)
        # print(dtag)
        pdb_block = pdb_grabber.get_bound_pdb_file(protein_code)
        try:
            # print(pdb_block)
            structure = gemmi.read_pdb_string(pdb_block)

            model = structure

            reference_models[dtag] = model
        except Exception as e:
            print(e)
            reference_models[dtag] = None

    return reference_models


def get_ligand(model):
    ligands = []

    for chain in model:
        # chain_ligands = chain.get_ligands()
        # for lig in chain_ligands:
        #     if lig.name == "LIG":
        #         ligands.append(lig)

        for res in chain:
            if res.name == "LIG":
                ligands.append(res)

    return ligands

def save_reference_models(pandda_dir,
                          reference_structures,
                          autobuild_structures,
                          ):
    processed_models_dir = pandda_dir / PANDDA_PROCESSED_DATASETS_DIR

    for dtag in autobuild_structures:
        model_dir = processed_models_dir / dtag / PANDDA_MODELLED_STRUCTURES_DIR
        model_path = model_dir / "fragalysis.pdb"
        if not model_path.exists():
            if dtag in reference_structures:
                structure = reference_structures[dtag]
                structure.write_pdb(str(model_path))
                logs.LOG[dtag]["remote_path"] = str(model_path)



def get_autobuild_rmsds(pandda_dir):
    # Get autobuilt models
    autobuild_structures = get_pandda_models(pandda_dir)

    # Get reference models
    project_code = pandda_dir_to_project_code(pandda_dir)
    reference_structures = get_reference_models(project_code)

    save_reference_models(pandda_dir,
                          reference_structures,
                          autobuild_structures,
                          )
    logs.LOG["number_of_reference_models"] = len(reference_structures)
    print("Number of reference models is: {}".format(len(reference_structures)))

    # print(autobuild_models)
    # print(reference_models)

    # Calculate distances between them
    records = []
    for dtag, reference_structure in reference_structures.items():
        record = {}
        record["dtag"] = dtag

        print("Analysing dataset: {}".format(dtag))

        if reference_structure is None:
            record["distance"] = None

            print("\t{} has not been hand built".format(dtag))

            continue

        if dtag not in autobuild_structures:
            record["distance"] = None

            print("\t{} has not been autobuilt".format(dtag))
            continue

        reference_model = reference_structure[0]
        autobuild_structure = autobuild_structures[dtag]
        autobuild_model = autobuild_structure[0]

        if len(reference_model) == 0:
            record["distance"] = None

            print("\t{} has an empty reference structure".format(dtag))
            continue


        autobuild_ligand_model = get_ligand(autobuild_model)[0]
        reference_ligand_models = get_ligand(reference_model)

        distances = []
        for reference_ligand_model in reference_ligand_models:
            distance = get_distance(reference_ligand_model,
                                    autobuild_ligand_model,
                                    )
            distances.append(distance)
        print("\tDistances are: {}".format(distances))
        record["distance"] = min(distances)

    records.append(record)

    # Table
    table = pd.DataFrame(records)

    return table

    # # Plot
    # plot_distance_distribution(table)

# def main():
#     mpro_summary = xcextracter('Mpro')
#     print(mpro_summary)
#     for col in mpro_summary.columns:
#         print(col)
#
#     # for index, row in mpro_summary.iterrows():
#     #     for x in row:
#     #         print(x)
#     #     exit()
#
#     # set up the pdb grabbing object
#     pdb_grabber = GetPdbData()
#
#     # use our selected code to pull out the pdb file (currently the file with ligand removed)
#     pdb_block = pdb_grabber.get_pdb_file(mol_id)
#
# if __name__ == "__main__":
#     main()
