import time
import subprocess
import re
import json

import numpy as np

import gemmi


class Rhofit:
    def __init__(self,
                 out_dir_path=None,
                 mtz_path=None,
                 ligand_path=None,
                 receptor_path=None,
                 ):
        env = "module load buster"
        ligand_fit_command = "rhofit"
        ligand_fit_args = "-m {mtz} -l {ligand} -p {receptor} -d {out_dir_path} -allclusters -use_2fofc"
        ligand_fit_args_formatted = ligand_fit_args.format(mtz=mtz_path,
                                                           ligand=ligand_path,
                                                           receptor=receptor_path,
                                                           out_dir_path=out_dir_path,
                                                           )
        self.command = "{env}; {ligand_fit_command} {args}".format(env=env,
                                                                   ligand_fit_command=ligand_fit_command,
                                                                   args=ligand_fit_args_formatted,
                                                                   )

    def __repr__(self):
        return self.command


class AutobuildingResultRhofit:
    def __init__(self, pandda_name, dtag, event_idx, rscc_string, stdout, stderr):
        self.pandda_name = pandda_name
        self.dtag = dtag
        self.event_idx = event_idx
        self.rscc = float(rscc_string)
        self.stdout = stdout
        self.stderr = stderr

    @staticmethod
    def null_result(event):
        return AutobuildingResultRhofit(event.pandda_name,
                                        event.dtag,
                                        event.event_idx,
                                        "0",
                                        "",
                                        "",
                                        )

    def to_json(self, path):
        record = {}
        record["pandda_name"] = self.pandda_name
        record["dtag"] = self.dtag
        record["event_idx"] = self.event_idx
        record["rscc"] = self.rscc
        record["stdout"] = self.stdout
        record["stderr"] = self.stderr

        json_string = json.dumps(record)
        with open(str(path), "w") as f:
            f.write(json_string)

    @staticmethod
    def from_json(path):
        with open(str(path), "r") as f:
            json_string = str(f.read())

        data = json.loads(json_string)

        return AutobuildingResultRhofit(pandda_name=data["pandda_name"],
                                        dtag=data["dtag"],
                                        event_idx=data["event_idx"],
                                        rscc_string=data["rscc"],
                                        stdout=data["stdout"],
                                        stderr=data["stderr"],
                                        )

    @staticmethod
    def from_output(rhofit_dir, pandda_name, dtag, event_idx, stdout="", stderr=""):
        rhofit_results_path = rhofit_dir / "results.txt"

        rscc_string = str(AutobuildingResultRhofit.parse_rhofit_log(rhofit_results_path))
        return AutobuildingResultRhofit(pandda_name, dtag, event_idx, rscc_string, stdout, stderr)

    @staticmethod
    def parse_rhofit_log(rhofit_results_path):
        regex = "(Hit_[^\s]+)[\s]+[^\s]+[\s]+[^\s]+[\s]+([^\s]+)"

        with open(str(rhofit_results_path), "r") as f:
            results_string = f.read()

        rscc_matches = re.findall(regex,
                                  results_string,
                                  )

        rsccs = {str(match_tuple[0]): float(match_tuple[1]) for match_tuple in rscc_matches}

        max_rscc = max(list(rsccs.values()))

        return max_rscc


class Elbow:
    def __init__(self,
                 autobuilding_dir,
                 ligand_smiles_path,
                 ):
        command = "module load phenix; cd {autobuilding_dir}; phenix.elbow {ligand_smiles_path} --output=\"{out}\""
        formatted_command = command.format(autobuilding_dir=autobuilding_dir,
                                           ligand_smiles_path=ligand_smiles_path,
                                           out="autobuilding_ligand",
                                           )

        self.command = formatted_command

    def __repr__(self):
        return self.command


class MapToMTZ:
    def __init__(self,
                 event_map_path,
                 output_path,
                 resolution,
                 col_f="FWT",
                 col_ph="PHWT",
                 gemmi_path="/dls/science/groups/i04-1/conor_dev/gemmi/gemmi",
                 ):
        command = "module load gcc/4.9.3; source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_clipper_no_mkl; {gemmi_path} map2sf {event_map_path} {output_path} {col_f} {col_ph} --dmin={resolution}"
        formatted_command = command.format(gemmi_path=gemmi_path,
                                           event_map_path=event_map_path,
                                           output_path=output_path,
                                           col_f=col_f,
                                           col_ph=col_ph,
                                           resolution=resolution,
                                           )
        self.command = formatted_command

    def __repr__(self):
        return self.command


class Strip:
    def __init__(self,
                 initial_receptor_path,
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
                distance = self.get_ligand_distance(ligand, coords)
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

    def get_ligand_mean_coords(self, residue):
        coords = []
        for atom in residue:
            pos = atom.pos
            coords.append([pos.x, pos.y, pos.z])

        coords_array = np.array(coords)
        mean_coords = np.mean(coords_array,
                              axis=0,
                              )
        return mean_coords

    def get_ligand_distance(self, ligand, coords):
        ligand_mean_coords = self.get_ligand_mean_coords(ligand)
        distance = np.linalg.norm(ligand_mean_coords - coords)
        return distance

    def remove_residue(self, chain, ligand):
        del chain[str(ligand.seqid)]


class Graft:
    def array_to_index(self, event_mtz):
        event_data = np.array(event_mtz, copy=False)
        array_to_index_map = {}
        for i in range(event_data.shape[0]):
            h = int(event_data[i, 0])
            k = int(event_data[i, 1])
            l = int(event_data[i, 2])
            array_to_index_map[i] = (h, k, l)

        return array_to_index_map

    def index_to_array(self, intial_mtz):
        event_data = np.array(intial_mtz, copy=False)
        index_to_array_map = {}
        for i in range(event_data.shape[0]):
            h = int(event_data[i, 0])
            k = int(event_data[i, 1])
            l = int(event_data[i, 2])
            index_to_array_map[(h, k, l)] = i

        return index_to_array_map

    def __init__(self,
                 initial_mtz_path,
                 new_mtz_path,
                 out_mtz_path,
                 ):
        intial_mtz = gemmi.read_mtz_file(str(initial_mtz_path))
        event_mtz = gemmi.read_mtz_file(str(new_mtz_path))

        array_to_index_map = self.array_to_index(intial_mtz)
        index_to_array_map = self.index_to_array(event_mtz)

        initial_mtz_data = np.array(intial_mtz, copy=False)
        event_mtz_data = np.array(event_mtz, copy=False)
        # print(initial_mtz_data.shape)
        # print(event_mtz_data.shape)

        # FWT
        initial_mtz_fwt = intial_mtz.column_with_label('FWT')
        # initial_mtz_fwt_index = initial_mtz_fwt.dataset_id
        initial_mtz_fwt_index = intial_mtz.column_labels().index("FWT")

        event_mtz_fwt = event_mtz.column_with_label('FWT')
        event_mtz_fwt_index = event_mtz.column_labels().index("FWT")

        # print("\t{}, {}".format(initial_mtz_data.shape, event_mtz_data.shape))
        # print(list(array_to_index_map.keys())[:10])
        # print(list(index_to_array_map.keys())[:10])

        skipped = 0
        for intial_array in range(initial_mtz_data.shape[0]):
            try:
                index = array_to_index_map[intial_array]
                event_array = index_to_array_map[index]
                initial_mtz_data[intial_array, initial_mtz_fwt_index] = event_mtz_data[event_array, event_mtz_fwt_index]
            except Exception as e:
                skipped = skipped + 1
                initial_mtz_data[intial_array, initial_mtz_fwt_index] = 0
        intial_mtz.set_data(initial_mtz_data)
        print("\tSkipped {} reflections".format(skipped))

        # PHWT
        initial_mtz_phwt_index = intial_mtz.column_labels().index("PHWT")

        event_mtz_phwt_index = event_mtz.column_labels().index("PHWT")

        skipped = 0
        for intial_array in range(initial_mtz_data.shape[0]):
            try:
                index = array_to_index_map[intial_array]
                event_array = index_to_array_map[index]
                initial_mtz_data[intial_array, initial_mtz_phwt_index] = event_mtz_data[
                    event_array, event_mtz_phwt_index]
            except Exception as e:
                skipped = skipped + 1
                initial_mtz_data[intial_array, initial_mtz_phwt_index] = 0
        intial_mtz.set_data(initial_mtz_data)
        print("\tCopied FWT from {} to {}".format(event_mtz_fwt_index, initial_mtz_fwt_index))
        print("\tCopied PHWT from {} to {}".format(event_mtz_phwt_index, initial_mtz_phwt_index))
        print("\tSkipper {} reflections".format(skipped))

        intial_mtz.write_to_file(str(out_mtz_path))



class QSub:
    def __init__(self,
                 command,
                 submit_script_path,
                 queue="low.q",
                 cores=1,
                 m_mem_free=10,
                 h_vmem=20,
                 ):
        self.command = command
        self.submit_script_path = submit_script_path
        self.queue = queue
        self.cores = cores
        self.m_mem_free = m_mem_free
        self.h_vmem = h_vmem

        with open(str(submit_script_path), "w") as f:
            f.write(command)

        chmod_proc = subprocess.Popen("chmod 777 {}".format(submit_script_path),
                                      shell=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      )
        chmod_proc.communicate()

        qsub_command = "qsub -q {queue} -pe smp {cores} -l m_mem_free={m_mem_free}G,h_vmem={h_vmem}G {submit_script_path}"
        self.qsub_command = qsub_command.format(queue=self.queue,
                                                cores=self.cores,
                                                m_mem_free=self.m_mem_free,
                                                h_vmem=self.h_vmem,
                                                submit_script_path=self.submit_script_path,
                                                )

    def __call__(self):
        print("\tCommand is: {}".format(self.command))
        print("\tQsub command is: {}".format(self.qsub_command))
        submit_proc = subprocess.Popen(self.qsub_command,
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       )
        stdout, stderr = submit_proc.communicate()

        proc_number = self.parse_process_number(str(stdout))

        time.sleep(10)

        while not self.is_finished(proc_number):
            time.sleep(10)

    def parse_process_number(self, string):
        regex = "[0-9]+"
        m = re.search(regex,
                      string,
                      )
        return m.group(0)

    def is_finished(self, proc_number):
        stat_proc = subprocess.Popen("qstat",
                                     shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     )
        stdout, stderr = stat_proc.communicate()

        if re.search(proc_number, str(stdout)):
            return False
        else:
            return True
