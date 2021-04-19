from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


def GetLabelFromFile(f, labels):
    for k, v in labels.items():
        if f.find(v) != -1:
            return k
    return -1


# Create and return a mean array of the three timeStamps plus an array with the number of samples in the action
# plus create an array with the label of the action
def CreateArrayWithoutAxis(df_0, df_1, df_2, label):
    df0 = df_0['TimeStamp'].to_numpy()
    df1 = df_1['TimeStamp'].to_numpy()
    df2 = df_2['TimeStamp'].to_numpy()

    meanTimeStamp = [0] * 100
    labels = [label] * 100
    for i in range(100):
        meanTimeStamp[i] = round(np.mean([df0[i], df1[i], df2[i]]))
    return meanTimeStamp, np.arange(100), labels


# Create and return an array with all the 6 axes inside an array
def CreateAxis(df_0, df_1, df_2):
    df0 = df_0[['X-Axis', 'Y-Axis', 'Z-Axis']].values
    df1 = df_1[['X-Axis', 'Y-Axis', 'Z-Axis']].values
    df2 = df_2[['X-Axis', 'Y-Axis', 'Z-Axis']].values
    allAxis = [0] * 100
    for i in range(100):
        allAxis[i] = np.concatenate((df0[i], df1[i], df2[i]), axis=None)
    return allAxis


# Create and return the complete DataFrame with all arrays pass in parameters
def CreateDataFrame(timeStamp, samples, axis, label):
    allData = pd.DataFrame({'TimeStamp': timeStamp, 'Sample No': samples,
                            'X-AxisA': [x[0] for x in Axis], 'Y-AxisA': [x[1] for x in Axis],
                            'Z-AxisA': [x[2] for x in Axis], 'X-AxisG': [x[3] for x in Axis],
                            'Y-AxisG': [x[4] for x in Axis], 'Z-AxisG': [x[5] for x in Axis],
                            'X-AxisM': [x[6] for x in Axis], 'Y-AxisM': [x[7] for x in Axis],
                            'Z-AxisM': [x[8] for x in Axis],
                            'Action': label})
    return allData


allfiles = [f for f in listdir("UMAFall_Dataset") if isfile(join("UMAFall_Dataset", f))]

features = "TimeStamp;Sample No;X-Axis;Y-Axis;Z-Axis;Sensor_Type;Sensor_ID\n"

# clean the beginning of the file
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

labels = {0: "Sitting_GettingUpOnAChair", 1: "Walking", 2: "Bending", 3: "backwardFall", 4: "forwardFall", 5: "lateralFall"}

AllFeatures = "TimeStamp,Sample No,X-AxisA,Y-AxisA,Z-AxisA,X-AxisG,Y-AxisG,Z-AxisG,X-AxisM,Y-AxisM,Z-AxisM,class\n"

allData = open('AllData.csv', 'w')
allData.write(AllFeatures)
allData.close()

# Write in the file 'AllData.csv' the all action with the accelerometer and gyroscope axis in the same row
for f in allfiles:
    name_file = join("UMAFall_Dataset", f)
    label = GetLabelFromFile(name_file, labels)
    if label != -1:
        with open(name_file) as csv_file:
            df = pd.read_csv(name_file, sep=';', header=0)
            df = df[df.Sensor_ID == 3]

            df_0 = pd.DataFrame(df[df.Sensor_Type == 0]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)
            df_1 = pd.DataFrame(df[df.Sensor_Type == 1]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)
            df_2 = pd.DataFrame(df[df.Sensor_Type == 2]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)

            if not df_2.empty:
                newTimeStamp, SampleNo, Labels = CreateArrayWithoutAxis(df_0, df_1, df_2, label)
                Axis = CreateAxis(df_0, df_1, df_2)

                df_final = CreateDataFrame(newTimeStamp, SampleNo, Axis, Labels)
                df_final['Action'] = np.where(df_final['Action'] >= 3, 'Fall', df_final['Action'])
                df_final['Action'] = np.where(df_final['Action'] != 'Fall', 'NotFall', df_final['Action'])

                df_final.to_csv(r'AllData.csv', mode='a', header=False, index=False, sep=',')
