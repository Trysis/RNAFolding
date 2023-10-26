import os
import re

# Pattern
r2_pattern = re.compile("R2=")

dirpath = "../out/bidirect"
fileindir = os.listdir(dirpath)
file_query = [fich[f.span()[1]:-8].split("_") for fich in fileindir \
              if (f := r2_pattern.search(fich)) is not None]

file_query = [[float(f[0]), f[1]] for f in file_query]
file_query = sorted(file_query, key=lambda k: k[0], reverse=True)

to_write = "id,r2\n"
for i in file_query:
    to_write += f"{i[1]},{i[0]}\n"

with open("best_seq.txt", "w") as file_out:
    file_out.write(to_write)

