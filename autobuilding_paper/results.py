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


class PanDDAResult:
    pass


class AutobuildResult:
    pass
