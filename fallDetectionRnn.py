import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import SimpleRNN
from tensorflow.keras import layers


Label = {0: 'fallen', 1: 'not_fallen'}

data_not = pd.read_csv('UMAFall_Dataset/UMAFall_Subject_01_ADL_Aplausing_1_2017-04-14_23-38-23.csv', sep = ';', header = 0)
#data_fall = pd.read_csv("UMAFall_Dataset/UMAFall_Subject_04_Fall_forwardFall_6_2016-06-13_13-23-05.csv", sep=';', header=0)
#data_test = pd.read_csv('UMAFall_Subject_01_ADL_Aplausing_2_2017-04-14_23-38-59.csv', sep = ';', header = 0)
labels = []
for i in range(len(data_not)):
    if (i%2) == 0:
        labels.append(1)
    else:
        labels.append(0)


model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])
model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

model.fit(np.array(data_not), np.array(labels), epochs=2)
print(model.summary())
