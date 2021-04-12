import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import SimpleRNN

Label = { 1:'fallen', 2: 'not_fallen'}



data_train = pd.read_csv('UMAFall_Subject_01_ADL_Aplausing_1_2017-04-14_23-38-23.csv', sep = ';', header = 0)
data_test = pd.read_csv('UMAFall_Subject_01_ADL_Aplausing_2_2017-04-14_23-38-59.csv', sep = ';', header = 0)
train_dataset = dataset['data_train']
test_dataset = dataset['sensor']

#la ligne qui suive sert à limiter les test que sur les 10 premier lignes.
pandas.options.display.max_rows = 10

model = Sequential()
model.add(SimpleRNN(units=16, input_shape=(10, 30), use_bias=False))
model.fit(train_dataset, test_dataset)
print(model.summary())