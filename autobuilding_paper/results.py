import json


class SystemTable:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def to_dict(self):
        return self.dictionary

    @staticmethod
    def from_json(file):
        dictionary = json.load(str(file))
        return SystemTable(dictionary)


class PanDDAResult:
    pass


class AutobuildResult:
    pass
