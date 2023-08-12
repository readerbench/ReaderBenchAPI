from enum import Enum, auto
import json
from typing import List


class TargetType(Enum):
    FLOAT = auto()
    INT = auto()
    STR = auto()

class Task:
    def __init__(self, type: TargetType=None, values: List[str]=None, obj=None):
        if obj is not None:
            for key, value in obj.items():
                if key == "type":
                    self.type = TargetType(value)
                else:
                    self.__dict__[key] = value
        else:
            self.binary = False
            if type is TargetType.INT:
                values = [int(val) for val in values]
                self.max = max(values)
                self.min = min(values)
                if self.max - self.min == 1 and len(set(values)) == 2:
                    self.binary = True
                    self.classes = [0, 1]
            elif type is TargetType.FLOAT:
                values = [float(val) for val in values]
                self.min = min(values)
                self.max = max(values)
            else:
                self.classes = list(sorted({str(val) for val in values}))
                self.index = {c: i for i, c in enumerate(self.classes)}
            self.type = type
            self.features: List[str] = []
        
    def convert_targets(self, values):
        result = []
        for val in values:
            if self.binary:
                result.append(int(val))
            elif self.type is TargetType.STR:
                result.append(self.index[val])
            else:
                result.append((float(val) - self.min) / (self.max - self.min))
        return result
    
    def convert_prediction(self, value):
        if self.binary:
            return value
        elif self.type is TargetType.STR:
            return self.classes[value]
        else:
            return value * (self.max - self.min) + self.min

    def __repr__(self):
        if self.type is TargetType.STR:
            info = f"{len(self.classes)} classes"
        else:
            info = f"[{self.min}, {self.max}]"
        return f"{self.type}: {info}"
    
    def save(self, filename: str):
        obj = dict(self.__dict__)
        obj["type"] = obj["type"].value
        with open(filename, "wt") as f:
            json.dump(obj, f)
    

def is_double(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True

def is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True
