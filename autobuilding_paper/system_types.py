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
class System:
    system_id: SystemID
    system_path: Path


@dataclass()
class Systems:
    systems: Dict[SystemID, System]

    @staticmethod
    def from_json(path: Path):
        with open(str(path), "w") as f:
            systems_dict = json.load(f)

        systems = {}
        for system_id_str, path_str in systems_dict.items():
            system_id = SystemID(system_id_str)
            path = Path(path_str)
            system = System(system_id=system_id,
                            system_path=path,
                            )
            systems[system_id] = system

        return Systems(systems)


@dataclass()
class SystemID:
    system_id: str
