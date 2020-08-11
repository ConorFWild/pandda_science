from pathlib import Path

import typing
import dataclasses

import json


class SystemTable:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def to_dict(self):
        return self.dictionary

    @staticmethod
    def from_json(file):
        with open(str(file), "r") as f:
            dictionary = json.load(f)
        return SystemTable(dictionary)


@dataclasses.dataclass()
class PanDDAResult:
    model_dir: Path
    out_dir: Path
    finished: bool
    events: typing.Dict

    def to_dict(self):
        return {"model_dir": str(self.model_dir),
                "out_dir": str(self.out_dir),
                "finished": self.finished,
                "events": self.events,
                }

    def to_json(self, path):
        with open(str(path), "w") as f:
            json.dump(self.to_dict(),
                      f,
                      )

    @staticmethod
    def from_json(file):
        with open(str(file), "r") as f:
            dictionary = json.load(f)
        return PanDDAResult(dictionary)


@dataclasses.dataclass()
class PanDDAResults:
    results: typing.Dict[str, PanDDAResult]

    @staticmethod
    def from_json(file):
        with open(str(file), "r") as f:
            dictionary = json.load(f)

        for key in dictionary:
            dictionary[key] = PanDDAResult[dictionary[key]]

        return PanDDAResults(dictionary)


class AutobuildResult:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    @staticmethod
    def from_json(file):
        with open(str(file), "r") as f:
            dictionary = json.load(f)
        return AutobuildResult(dictionary)
