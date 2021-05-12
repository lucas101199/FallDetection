from os import listdir
from os.path import isfile, join


folder = 'FallUser'
allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
f = open('raw_data.txt', 'a+')
for file in allfiles:
    with open(join(folder, file), 'r') as fi:
        linetowrite = fi.readlines()[0] + '0\n'
        f.write(linetowrite)
