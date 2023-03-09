import json
from os import scandir
from string import ascii_uppercase, digits, punctuation
import pandas as pd

valid_chars = ascii_uppercase + digits + punctuation + " \t\n"

csv_files = sorted(
    (f for f in scandir(r"D:\study\dataset\sroie-2019\raw\box") if f.name.endswith("csv")), key=lambda f: f.name
)
# json_files = sorted(
#     (f for f in scandir(r"D:\study\dataset\sroie-2019\raw\box") if f.name.endswith("json")), key=lambda f: f.name
# )

for f in csv_files:
    with open(f, "r", encoding="utf-8") as fo:
        handoi = []
        for line_no, line in enumerate(fo, start=1):
            # import pdb; pdb.set_trace()
            entries = line.split(",", maxsplit=8)
            for c in entries[-1]:
                if not c in valid_chars:
                    print(f"Invalid char {repr(c)} in {f.name} on line {line_no}")
            handoi.append(entries)
        df = pd.DataFrame(handoi)
        df.to_csv('New folder/' + f.name, index=False, header=None)

# for f in json_files:
#     with open(f, "r", encoding="utf-8") as fo:
#         content = json.load(fo)
#         for key in content:
#             for c in content[key]:
#                 if not c in valid_chars:
#                     print(f"Invalid char {repr(c)} in {f.name} on line {line_no}")