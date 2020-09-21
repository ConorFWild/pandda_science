STRIPPED_RECEPTOR_FILE = "stripped_receptor.pdb"
LIGAND_FILE = "autobuilding_ligand.cif"
EVENT_MTZ_FILE = "event.mtz"
GRAFTED_MTZ_FILE = "grafted.mtz"
RHOFIT_DIR = "rhofit"
RHOFIT_EVENT_DIR = "rhofit_{}"
RHOFIT_NORMAL_DIR = "rhofit_normal"
RHOFIT_RESULTS_FILE = "results.txt"
RHOFIT_RESULT_JSON_FILE = "result.json"
RHOFIT_NORMAL_RESULT_JSON_FILE = "result_normal.json"
RHOFIT_BEST_MODEL_FILE = "best.pdb"
RSCC_TABLE_FILE = "rscc_table.csv"
RHOFIT_HIT_REGEX = "(Hit_[^\s]+)[\s]+[^\s]+[\s]+[^\s]+[\s]+([^\s]+)"
RHOFIT_CLUSTER_BUILD_REGEX = "Hit_([^_]+)_[^_]+_([^_]+).pdb"

BUILD_DIR_PATTERN = "{pandda_name}_{dtag}_{event_idx}"


CUMULATIVE_HITS_PLOT_FILE = "cumulative_hits.png"


PANDDA_ANALYSES_DIR = "analyses"
PANDDA_ANALYSE_EVENTS_FILE = "pandda_analyse_events.csv"
PANDDA_PROCESSED_DATASETS_DIR = "processed_datasets"
PANDDA_MODELLED_STRUCTURES_DIR = "modelled_structures"
PANDDA_LIGAND_FILES_DIR = "ligand_files"
PANDDA_PDB_FILE = "{}-pandda-input.pdb"
PANDDA_MTZ_FILE = "{}-pandda-input.mtz"
PANDDA_INSPECT_EVENTS_PATH = "pandda_inspect_events.csv"
PANDDA_EVENT_MAP_FILE = "{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4"
PANDDA_EVENT_MODEL = "{}-pandda-model.pdb"


PANDDA_Z_MAP_FILE = "{dtag}-z_map.native.ccp4"

AUTOBUILD_EVENT_MTZ = "{dtag}_{event_idx}.ccp4"
AUTOBUILD_ENV = "module load buster"
AUTOBUILD_COMMAND = "rhofit"
AUTOBUILD_ARGS = "-m {mtz} -l {ligand} -d {out_dir_path} -allclusters -use_2fofc"