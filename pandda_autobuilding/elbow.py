import subprocess

def elbow(autobuilding_dir, ligand_smiles_path):
    command = "module load phenix; cd {autobuilding_dir}; phenix.elbow {ligand_smiles_path} --output=\"{out}\""
    formatted_command = command.format(autobuilding_dir=autobuilding_dir,
                                       ligand_smiles_path=ligand_smiles_path,
                                       out="ligand",
                                       )

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    return stdout, stderr
