import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.svm import SVC
import math
import numpy as np


def CreateDataset(TimeSteps, df):
    size = math.ceil(len(df)/TimeSteps)
    array_axis = np.empty([size, TimeSteps, 6])
    j = 0
    for i in range(0, len(df), TimeSteps):
        array_axis[j] = df.loc[i:i+TimeSteps-1, ['X-AxisA', 'Y-AxisA', 'Z-AxisA', 'X-AxisG', 'Y-AxisG', 'Z-AxisG']].values
        j += 1
    return array_axis

df = pd.read_csv('AllData.csv', sep=';', header=0)

labels = df.pop('Action')
features = df.values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
array = CreateDataset(60, df)
print(array)
