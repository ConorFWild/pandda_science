import json


class Log:
    def __init__(self, dictionary=None):
        if dictionary:
            self.dict = dictionary
        else:
            self.dict = {}

    def __getitem__(self, item):
        return self.dict[item]

    def __setitem__(self, key, value):
        self.dict[key] = value

    def to_json(self, path):
        with open(str(path), "w") as f:
            json.dump(self.dict, f)

    @staticmethod
    def from_json(path):
        with open(str(path), "r") as f:
            dictionary = json.load(f)
        return Log(dictionary=dictionary)

LOG = Log()