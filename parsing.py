from os import listdir
from os.path import isfile, join
import pandas as pd


def GetLabelFromFile(f, labels):
    for k, v in labels.items():
        if f.find(v) != -1:
            return k
    return -1


allfiles = [f for f in listdir("UMAFall_Dataset") if isfile(join("UMAFall_Dataset", f))]

features = "TimeStamp;Sample No;X-Axis;Y-Axis;Z-Axis;Sensor_Type;Sensor_ID\n"

for f in allfiles:
    file = open(join("UMAFall_Dataset", f), "r")
    lines = file.readlines()
    file.close()
    if lines[0] != features:
        output = [features]
        new_file = open(join("UMAFall_Dataset", f), "w+")
        for i in range(len(lines)):
            if lines[i].find("%") == 0 or len(lines[i].split()) == 0:
                pass
            else:
                output.append(lines[i])
        new_file.writelines(output)
        new_file.close()

labels = {0: "Aplausing", 1: "HandsUp", 2: "MakingACall", 3: "OpeningDoor",
        4: "Sitting_GettingUpOnAChair", 5: "Walking", 6: "Bending", 7: "Hopping",
        8: "Jogging", 9: "LyingDown", 10: "GoDownstairs", 11: "GoUpstairs", 12: "Fall"}

features_action = features[:-1] + ';Action\n'
allData = open('AllData.csv', 'w')
allData.write(features_action)
allData.close()

for f in allfiles:
    name_file = join("UMAFall_Dataset", f)
    label = GetLabelFromFile(name_file, labels)
    with open(name_file) as csv_file:
        df = pd.read_csv(name_file, sep=';', header=0)
        df = df[df.Sensor_ID == 3]
        action = [label] * len(df.index)
        df["Action"] = action
        df.to_csv(r'AllData.csv', mode='a', header=False, index=False, sep=';')

#print(data)
