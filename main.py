from os import listdir
from os.path import isfile, join

import numpy as np

from segmentation import sliding_window
from preprocess import ChangeData
from features import WriteFeatures
from sklearn.preprocessing import MinMaxScaler


def normalizeData(file):
    f = open(file, 'r')
    lines = f.readlines()
    data_normalize = []
    for line in lines:
        features = []
        for l in line.split(' '):
            features.append(int(l))
        data_normalize.append(features)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data_normalize)


folder = 'NotFall'
allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
filetxt = 'raw_data.txt'
for file in allfiles:
    df = ChangeData(file, 3)  # preprocess
    sliding_window(df, 2)  # segmentation

WriteFeatures(filetxt)  # features extraction
norm_Data = normalizeData('features.txt')
np.savetxt('features_norm.txt', norm_Data, fmt='%1.4e')
