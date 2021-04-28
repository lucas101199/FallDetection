"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import random


# Class all files within a folder into different categories
# Return a dictionary with as key the name of the class and as value a list of all files in the class
def ClassFilesIntoCategory(folder):
    category = {'Sitting_GettingUpOnAChair': [], 'Walking': [], 'Bending': [], 'backwardFall': [],
                'forwardFall': [], 'lateralFall': []}
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
                            'Class': label})
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


# Write in the file 'AllData.csv' all fall data with the accelerometer and gyroscope in the same row
def WriteInFileFall(files, NameFile, folder):
    for f in files:
        name_file = join(folder, f)
        label = 'Fall'
        with open(name_file):
            df = pd.read_csv(name_file, sep=';', header=0)
            df = df[df.Sensor_ID == 3]

            df_0 = pd.DataFrame(df[df.Sensor_Type == 0]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)
            df_1 = pd.DataFrame(df[df.Sensor_Type == 1]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)
            df_2 = pd.DataFrame(df[df.Sensor_Type == 2]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)

            if not df_2.empty:
                newTimeStamp, SampleNo, Labels = CreateArrayWithoutAxis(df_0, df_1, df_2, label)
                Axis = CreateAxis(df_0, df_1, df_2)

                df_final = CreateDataFrame(newTimeStamp, SampleNo, Axis, Labels)

                df_final.to_csv(NameFile, mode='a', header=False, index=False, sep=',')


def WriteInFileNotFall(file, NameFile, folder):
    name_file = join(folder, file)
    label = 'NotFall'
    with open(name_file):
        df = pd.read_csv(name_file, sep=';', header=0)
        df = df[df.Sensor_ID == 3]

        df_0 = pd.DataFrame(df[df.Sensor_Type == 0]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)
        df_1 = pd.DataFrame(df[df.Sensor_Type == 1]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)
        df_2 = pd.DataFrame(df[df.Sensor_Type == 2]).drop(['Sensor_Type', 'Sensor_ID'], axis=1).head(100)

        if not df_2.empty:
            newTimeStamp, SampleNo, Labels = CreateArrayWithoutAxis(df_0, df_1, df_2, label)
            Axis = CreateAxis(df_0, df_1, df_2)

            df_final = CreateDataFrame(newTimeStamp, SampleNo, Axis, Labels)

            df_final.to_csv(NameFile, mode='a', header=False, index=False, sep=',')


# Get the file list for a given category, select randomly a file and then delete it for the list
def ChooseFile(category, i):
    sizeDict = len(category)
    name_category = list(category.keys())[i % sizeDict]
    listFile = category[name_category]
    j = 1
    while len(listFile) == 0:
        name_category = list(category.keys())[(i+j) % sizeDict]
        listFile = category[name_category]
        j += 1
    file = random.choice(listFile)
    listFile.remove(file)
    return file


# Get all the files containing the fall data
def GetFileFall(dict):
    fallFiles = []
    for k, v in dict.items():
        if 'Fall' in k:
            for file in v:
                fallFiles.append(file)
    return fallFiles


folder = "UMAFall_Dataset"
allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

print("Clean up files...\n")
CleanUpFiles(allfiles)

print("Class files into category...\n")
category = ClassFilesIntoCategory(folder)
#print(len(category['lateralFall']))
AllFeatures = "TimeStamp,Sample No,X-AxisA,Y-AxisA,Z-AxisA,X-AxisG,Y-AxisG,Z-AxisG,X-AxisM,Y-AxisM,Z-AxisM,Class\n"

allData = open('AllData.csv', 'w')
allData.write(AllFeatures)
allData.close()

# Get the files containing the fall data
fallFiles = GetFileFall(category)

print("Write the fall data in the file...\n")
WriteInFileFall(fallFiles, 'AllData.csv', folder)

print("Write the not fall data in the file...\n")
df = pd.read_csv('AllData.csv', header=0)
fallData = pd.DataFrame(df[df.Class == 'Fall']).shape[0]
notFallData = 0
i = 0
while fallData >= notFallData:
    df = pd.read_csv('AllData.csv', header=0)
    file = ChooseFile(category, i)
    WriteInFileNotFall(file, 'AllData.csv', folder)
    notFallData = pd.DataFrame(df[df.Class == 'NotFall']).shape[0]
    i += 1
"""
