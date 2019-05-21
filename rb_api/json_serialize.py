import json
from enum import Enum

class JsonSerialize:

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.value if isinstance(o, Enum) else o.__dict__,
                          sort_keys=True, indent=4)
