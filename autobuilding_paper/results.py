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
    def from_dict(dictionary):
        return PanDDAResult(model_dir=dictionary["model_dir"],
                            out_dir=dictionary["out_dir"],
                            finished=dictionary["finished"],
                            events=dictionary["events"],
                            )

    @staticmethod
    def from_json(file):
        with open(str(file), "r") as f:
            dictionary = json.load(f)
        return PanDDAResult.from_dict(dictionary)


@dataclasses.dataclass()
class PanDDAResults:
    results: typing.Dict[str, PanDDAResult]

    @staticmethod
    def from_json(file):
        with open(str(file), "r") as f:
            dictionary = json.load(f)

        for key in dictionary:
            dictionary[key] = PanDDAResult.from_dict(dictionary[key])

        return PanDDAResults(dictionary)


    def to_dict(self):
        return self.results

@dataclasses.dataclass()
class AutobuildResult:
    builds: typing.Dict[typing.Tuple[str], typing.Dict]

    @staticmethod
    def from_dict(dictionary):

        builds = {}
        for dtag in dictionary:
            for event_idx in dictionary[dtag]:
                if dtag not in builds:
                    builds[dtag] = {}

                builds[dtag][event_idx] = dictionary[dtag][event_idx]

        return AutobuildResult(builds)


@dataclasses.dataclass()
class AutobuildResults:
    results: typing.Dict[str, AutobuildResult]

    @staticmethod
    def from_json(file):
        with open(str(file), "r") as f:
            string = str(f.read())
            dictionary = json.loads(string)

        results = {}
        for system in dictionary:
            results[system] = AutobuildResult.from_dict(dictionary[system])


        return AutobuildResults(results)

    def __iter__(self):
        for dtag in self.results:
            yield dtag

    def __getitem__(self, item):
        return self.results[item]