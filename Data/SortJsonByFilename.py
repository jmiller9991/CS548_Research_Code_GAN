import json
import os
from sys import argv

with open(argv[1], "r") as f:
    json_data = json.load(f)

json_data.sort(key=lambda d: d["localFacePath"])

filename, ext = os.path.splitext(argv[1])
with open(f"{filename}_sorted{ext}", "w") as f:
    json.dump(json_data, f, indent=4)
