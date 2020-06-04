import numpy as np

import gemmi


def get_ligand_mean_coords(residue):
    coords = []
    for atom in residue:
        pos = atom.pos
        coords.append([pos.x, pos.y, pos.z])

    coords_array = np.array(coords)
    mean_coords = np.mean(coords_array,
                          axis=0,
                          )
    return mean_coords


def get_ligand_distance(ligand, coords):
    ligand_mean_coords = get_ligand_mean_coords(ligand)
    distance = np.linalg.norm(ligand_mean_coords - coords)
    return distance


def remove_residue(chain, ligand):
    del chain[str(ligand.seqid)]


def strip_protein(initial_receptor_path,
                  coords,
                  receptor_path,
                  ):
    # Load protein
    structure = gemmi.read_structure(str(initial_receptor_path))
    receptor = structure[0]

    # Strip nearby residues
    remove_ids = []
    for chain in receptor:
        ligands = chain.get_ligands()
        for ligand in ligands:
            distance = get_ligand_distance(ligand, coords)
            if distance < 10:
                print("\t\tWill strip res {}. Mean distance {} from event!".format(ligand, distance))
                remove_ids.append(str(ligand.seqid))

    for chain in receptor:
        deletions = 0
        for i, residue in enumerate(chain):
            if str(residue.seqid) in remove_ids:
                print("\t\tStripping {}".format(residue))
                del chain[i - deletions]
                deletions = deletions + 1

    # Save
    structure.write_pdb(str(receptor_path))
