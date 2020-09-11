from __future__ import annotations

from dataclasses import dataclass
import typing

import re
from pathlib import Path
from subprocess import Popen, PIPE

import pandas as pd

import gemmi

# constants
RSCC_TABLE_FILE = "/dls/labxchem/data/2019/lb18145-130/processing/analysis/RSCC/table.csv"
MODELS_DIR = "/dls/labxchem/data/2019/lb18145-130/processing/analysis/RSCC"
MAPS_DIR = "/dls/labxchem/data/2019/lb18145-130/processing/analysis/model_building"
EVENT_MAP_FILE = "{dtag}-event_{event_idx}_1-BDC_{}_map.native.ccp4"

FOFC_MAP_PATTERN = "fofc.map"
TWOFOFC_MAP_PATTERN = "2fofc.map"
EVENT_MAP_PATTERN = "[^\-]+\-[^\-]+\-event_[0-9]+_1\-BDC_[^_]+_map.native.ccp4"
MTZ_PATTERN = "dimple.mtz"

PHENIX_ENV = "module load phenix"
PHENIX_MODEL_VS_MAP = "phenix.model_vs_map"
PHENIX_MAP_MODEL_CC = "phenix.map_model_cc"
CC_PER_RESIDUE_LOG_FILE = "cc_per_residue.log"
# CC_PER_RESIDUE_LOG_RSCC_PATTERN = "[a-zA-Z]+\s+LIG\s+[0-9]+\s([^\s]+)"
CC_PER_RESIDUE_LOG_RSCC_PATTERN = "[a-zA-Z]+\s+[^\s]\s+1501\s([^\s]+)"

@dataclass()
class Dtag:
    dtag: str

    @staticmethod
    def from_rscc_path(path: Path):
        pattern = "RSCC_([^\.]+)\.pdb"
        string = str(path)
        matches = re.findall(pattern,
                        string,
                        )

        first_match = matches[0]
        dtag = first_match
        return Dtag(dtag)

    def __hash__(self):
        return hash(self.dtag)


@dataclass()
class Models:
    models: typing.Dict[Dtag, Path]

    def __iter__(self):
        for dtag in self.models:
            yield dtag

    def __getitem__(self, item):
        return self.models[item]

    @staticmethod
    def from_dir(directory: Path):
        models = {}

        model_paths = list(directory.glob("*"))
        print("\tGot {} model paths".format(len(model_paths)))
        for model_path in model_paths:
            dtag = Dtag.from_rscc_path(model_path)
            models[dtag] = model_path

        return Models(models)


@dataclass()
class Maps:
    maps: typing.Dict[Dtag, Path]

    def __iter__(self):
        for dtag in self.maps:
            yield dtag

    def __getitem__(self, item):
        return self.maps[item]

    @staticmethod
    def from_dir(directory: Path, pattern: str):

        maps = {}
        dataset_dirs = list(directory.glob("*"))
        print("\tGot {} model paths".format(len(dataset_dirs)))
        for dataset_path in dataset_dirs:
            dtag = Dtag(dataset_path.name)
            for dataset_dir_file in dataset_path.glob("*"):
                matches = re.findall(pattern,
                                     str(dataset_dir_file))
                if len(matches) != 0:
                    xmap_file = dataset_dir_file
                    maps[dtag] = xmap_file

        return Maps(maps)


@dataclass()
class Model:
    pass


@dataclass()
class Map:
    pass


@dataclass()
class RSCC:
    rscc: float

    @staticmethod
    def from_model_and_map(model: Path, xmap: Path, resolution: Resolution):
        command = f"{PHENIX_ENV}; {PHENIX_MAP_MODEL_CC} {model} {xmap} resolution={resolution.resolution} cc_per_residue=True"

        process = Popen(command,
                        shell=True,
                        stdout=PIPE,
                        )

        stdout, stderr = process.communicate()

        rscc = RSCC.parse_log()

        return RSCC(rscc)

    @staticmethod
    def parse_stdout(stdout: str):
        with open(CC_PER_RESIDUE_LOG_FILE, "r") as f:
            string = f.read()

        matches = re.findall(CC_PER_RESIDUE_LOG_RSCC_PATTERN,
                             string)

        first_match = next(matches)
        first_match_capture_group = first_match.group(1)
        rscc = float(first_match_capture_group)

        return rscc

    @staticmethod
    def parse_log():
        with open(CC_PER_RESIDUE_LOG_FILE, "r") as f:
            string = f.read()

        matches = re.findall(CC_PER_RESIDUE_LOG_RSCC_PATTERN,
                             string)

        first_match = matches[0]
        first_match_capture_group = first_match.group(1)
        rscc = float(first_match_capture_group)

        return rscc


@dataclass()
class RSCCs:
    rsccs: typing.Dict[Dtag, RSCC]

    @staticmethod
    def from_models_and_maps(models: Models, maps: Maps, mtzs: MTZs):
        rsccs: typing.Dict[Dtag, RSCC] = {}
        for dtag in models:
            print("\tLooking at dtag: {}".format(dtag))
            model = models[dtag]
            xmap = maps[dtag]
            mtz = mtzs[dtag]
            resolution = Resolution.from_mtz(mtz)
            rsccs[dtag] = RSCC.from_model_and_map(model,
                                                  xmap,
                                                  resolution,
                                                  )
            print("\t\tRSCC is {}".format(rsccs[dtag]))

        return RSCCs(rsccs)

    def to_table(self):
        records = []
        for dtag in self.rsccs:
            record = {"dtag": dtag.dtag,
                      "rscc": self.rsccs[dtag].rscc,
                      }
            records.append(record)

        return pd.DataFrame(records)


@dataclass()
class MTZs:
    mtzs: typing.Dict[Dtag, Path]

    @staticmethod
    def from_dir(pandda_dir: Path, mtz_pattern: str):
        mtzs = {}
        dataset_dirs = list(pandda_dir.glob("*"))
        print("\tGot {} dataset dirs".format(len(dataset_dirs)))

        for dataset_dir in dataset_dirs:
            dtag = Dtag(dataset_dir.name)
            dataset_dir_files = dataset_dir.glob("*")
            for dataset_dir_file in dataset_dir_files:
                matches = re.findall(mtz_pattern, str(dataset_dir_file))

                if len(matches) != 0:
                    mtz_file = dataset_dir_file
                    mtzs[dtag] = mtz_file
                    continue

        return MTZs(mtzs)

    def __getitem__(self, item):
        return self.mtzs[item]


@dataclass()
class Resolution:
    resolution: float

    @staticmethod
    def from_mtz(mtz_file: Path):
        mtz = gemmi.read_mtz_file(str(mtz_file))
        resolution = mtz.resolution_high()

        return Resolution(resolution)


def main():
    print("Geting models...")
    models: Models = Models.from_dir(Path(MODELS_DIR))
    print("\tGot {} models".format(len(models.models)))

    print("Getting mtzs...")
    mtzs: MTZs = MTZs.from_dir(Path(MAPS_DIR),
                               MTZ_PATTERN,
                               )
    print("\tGot {} mtzs".format(len(mtzs.mtzs)))

    # Event maps
    print("Getting event maps...")
    maps: Maps = Maps.from_dir(Path(MAPS_DIR),
                               EVENT_MAP_PATTERN,
                               )
    print("\tGot {} maps".format(len(maps.maps)))
    rsccs_event_maps = RSCCs.from_models_and_maps(models,
                                                  maps,
                                                  mtzs,
                                                  )
    rsccs_event_maps_table = rsccs_event_maps.to_table()
    rsccs_event_maps_table["type"] = "event_map"

    # FOFC maps
    maps: Maps = Maps.from_dir(Path(MAPS_DIR),
                               FOFC_MAP_PATTERN,
                               )
    rsccs_event_maps = RSCCs.from_models_and_maps(models,
                                                  maps,
                                                  mtzs,
                                                  )
    rsccs_fofc_table = rsccs_event_maps.to_table()
    rsccs_fofc_table["type"] = "FOFC"

    # 2FOFC maps
    maps: Maps = Maps.from_dir(Path(MAPS_DIR),
                               TWOFOFC_MAP_PATTERN,
                               )
    rsccs_event_maps = RSCCs.from_models_and_maps(models,
                                                  maps,
                                                  mtzs,
                                                  )
    rsccs_2fofc_table = rsccs_event_maps.to_table()
    rsccs_2fofc_table["type"] = "2FOFC"

    # output
    table = pd.concat([rsccs_event_maps_table, rsccs_fofc_table, rsccs_2fofc_table])
    table.to_csv(str(RSCC_TABLE_FILE))


if __name__ == "__main__":
    main()
