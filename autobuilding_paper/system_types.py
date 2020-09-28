from __future__ import annotations

from typing import *

from dataclasses import dataclass

import os
import argparse
import re
import subprocess
import shutil
import json
from pathlib import Path
from pprint import PrettyPrinter


@dataclass()
class SystemID:
    system_id: str

    def __hash__(self):
        return hash(self.system_id)

    def to_python(self):
        return self.system_id


@dataclass()
class System:
    system_id: SystemID
    system_path: Path


@dataclass()
class Systems:
    systems: Dict[SystemID, System]

    @staticmethod
    def from_json(path: Path):
        with open(str(path), "r") as f:
            systems_dict = json.load(f)

        print(systems_dict)

        systems = {}
        for system_id_str, path_str in systems_dict.items():
            if not path_str:
                continue

            system_id = SystemID(system_id_str)
            path = Path(path_str)
            system = System(system_id=system_id,
                            system_path=path,
                            )
            systems[system_id] = system

        return Systems(systems)

    def __getitem__(self, item):
        return self.systems[item]

    def __iter__(self):
        for system_id in self.systems:
            yield system_id

    def to_dict(self):
        return self.systems