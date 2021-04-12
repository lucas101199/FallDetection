from os import listdir
from os.path import isfile, join
import pandas as pd


allfiles = [f for f in listdir("UMAFall_Dataset") if isfile(join("UMAFall_Dataset", f))]
clean_file = []

for f in allfiles:
    file = open(join("UMAFall_Dataset", f), "r")
    lines = file.readlines()
    file.close()
    output = []
    new_file = open(join("UMAFall_Dataset", f), "w+")
    for i in range(len(lines)):
        if lines[i].find("%") == 0 or len(lines[i].split()) == 0:
            pass
        else:
            output.append(lines[i])
    new_file.writelines(output)
    new_file.close()

"""
for f in allfiles:
    with open(join("UMAFall_Dataset", f)) as csv_file:
        file = pd.read_csv(join("UMAFall_Dataset", f))
"""
