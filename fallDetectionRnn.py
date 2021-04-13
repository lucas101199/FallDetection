import keras as keras
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import StandardScaler


Label = {0 : 'fallen', 1 : 'not_fallen'}

data_not = pd.read_csv('UMAFall_Dataset/UMAFall_Subject_01_ADL_Aplausing_1_2017-04-14_23-38-23.csv', sep=';', header=0)
# data_fall = pd.read_csv("UMAFall_Dataset/UMAFall_Subject_04_Fall_forwardFall_6_2016-06-13_13-23-05.csv", sep=';', header=0)
# data_test = pd.read_csv('UMAFall_Subject_01_ADL_Aplausing_2_2017-04-14_23-38-59.csv', sep = ';', header = 0)
labels = []
for i in range(3):
    if (i % 2) == 0 :
        labels.append(1)
    else :
        labels.append(0)
features = data_not.values
scaler = StandardScaler()
labels = np.array(labels)
features3dim = [[[0, 1], [5, 6]], [[10, 11], [15, 16]], [[20, 21], [25, 26]]]
j = 1
l = 0
"""
for i in range(2):
    if i % 150 == 0:
        l = l + 1
    features3dim.append()
    j = j + 1
    if j % 150 == 0:
        j = 1
"""

features3dim = np.array(features3dim)
print(features3dim.shape)

model = tf.keras.Sequential([
    keras.layers.LSTM(2, input_shape=(2, 2)),
    keras.layers.Dense(1)
])

model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

model.fit(features3dim, labels, epochs=2, batch_size=1)
print(model.output_shape)
