import glob
from datetime import datetime


allfiles = [f for f in glob.glob("*.csv") if f != 'eponge_2_L.csv']
for file in allfiles:
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    for i in range(1, len(lines)):
        l = lines[i].split(',')
        timestamp = l[0]
        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
        lines[i] = lines[i].replace(timestamp, str(dt.timestamp()))
    fw = open(file, 'w')
    fw.writelines(lines)
    fw.close()
