import os
import re

# Pattern
r2_DMS_pattern = re.compile("R2=.*len.*DMS")
r2_2A3_pattern = re.compile("R2=.*len.*2A3")

# Path
modelname = "spot"
dirpath = "./out/" + modelname
fileindir = os.listdir(dirpath)

file_DMS_query = [fich[f.span()[0]+3:-8].split("_") for fich in fileindir \
                 if (f := r2_DMS_pattern.search(fich)) is not None]

file_2A3_query = [fich[f.span()[0]+3:-8].split("_") for fich in fileindir \
                 if (f := r2_2A3_pattern.search(fich)) is not None]

# DMS
file_DMS_query = [[float(f[0]), f[2]] for f in file_DMS_query]
file_DMS_query = sorted(file_DMS_query, key=lambda k: k[0], reverse=True)
# 2A3
file_2A3_query = [[float(f[0]), f[2]] for f in file_2A3_query]
file_2A3_query = sorted(file_2A3_query, key=lambda k: k[0], reverse=True)

# DMS
to_write_DMS = "id,r2_DMS\n"
for i in file_DMS_query:
    to_write_DMS += f"{i[1]},{i[0]}\n"

with open(f"{modelname}_id_r2_DMS.txt", "w") as file_out:
    file_out.write(to_write_DMS)

# 2A3
to_write_2A3 = "id,r2_2A3\n"
for i in file_2A3_query:
    to_write_2A3 += f"{i[1]},{i[0]}\n"

with open(f"{modelname}_id_r2_2A3.txt", "w") as file_out:
    file_out.write(to_write_2A3)

