from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


def GetLabelFromFile(f, labels):
    for k, v in labels.items():
        if f.find(v) != -1:
            return k
    return -1


# Class all files within a folder into different categories
# Return a dictionary with as key the name of the class and as value a list of all files in the class
def ClassFilesIntoCategory(folder):
    category = {'Aplausing': [], 'HandsUp': [], 'MakingACall': [], 'OpeningDoor': [], 'Sitting_GettingUpOnAChair': [],
                'Walking': [], 'Bending': [], 'Hopping': [], 'Jogging': [], 'LyingDown': [], 'backwardFall': [],
                'forwardFall': [], 'lateralFall': [], 'GoDownstairs': [], 'GoUpstairs': []}
    for f in listdir(folder):
        for k, v in category.items():
            if f.find(k) != -1:
                v.append(f)
    return category


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
                            'X-AxisA': [x[0] for x in axis], 'Y-AxisA': [x[1] for x in axis],
                            'Z-AxisA': [x[2] for x in axis], 'X-AxisG': [x[3] for x in axis],
                            'Y-AxisG': [x[4] for x in axis], 'Z-AxisG': [x[5] for x in axis],
                            'X-AxisM': [x[6] for x in axis], 'Y-AxisM': [x[7] for x in axis],
                            'Z-AxisM': [x[8] for x in axis],
                            'Action': label})
    return allData


def CleanUpFiles(files):
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


# Write in the file 'AllData.csv' the all action with the accelerometer and gyroscope axis in the same row
def WriteInFile(files, NameFile, folder):
    for f in files:
        name_file = join(folder, f)
        label = GetLabelFromFile(name_file, category)
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
                    df_final['Action'] = np.where(df_final['Action'] >= 4, 'Fall', df_final['Action'])
                    df_final['Action'] = np.where(df_final['Action'] != 'Fall', 'NotFall', df_final['Action'])

                    df_final.to_csv(NameFile, mode='a', header=False, index=False, sep=',')


folder = "UMAFall_Dataset"
allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
CleanUpFiles(allfiles)

category = ClassFilesIntoCategory(folder)

AllFeatures = "TimeStamp,Sample No,X-AxisA,Y-AxisA,Z-AxisA,X-AxisG,Y-AxisG,Z-AxisG,X-AxisM,Y-AxisM,Z-AxisM,Class\n"

allData = open('AllData.csv', 'w')
allData.write(AllFeatures)
allData.close()

WriteInFile(allfiles, 'AllData.csv', folder)
