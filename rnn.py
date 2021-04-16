import math
import numpy as np
import pandas as pd
import preprocessing as preprocessing
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Bidirectional
from keras.utils.np_utils import to_categorical
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from sklearn.svm import SVC
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.metrics import confusion_matrix


# Create a dataset from a DataFrame of size [samples, timesteps, features] and size [samples, action] for the label
def CreateDataset(TimeSteps, df):
    size = math.ceil(len(df) / TimeSteps)
    features = np.empty([size, TimeSteps, 9])
    j = 0
    label = []
    for i in range(0, len(df), TimeSteps):
        features[j] = df.loc[i:i + TimeSteps - 1,
                      ['X-AxisA', 'Y-AxisA', 'Z-AxisA', 'X-AxisG', 'Y-AxisG', 'Z-AxisG', 'X-AxisM', 'Y-AxisM', 'Z-AxisM']].values
        action = df.loc[i:i + TimeSteps - 1, ['Action']].values
        action = action.flatten()
        label.append(np.bincount(action).argmax())
        j += 1
    return features, label


df = pd.read_csv('AllData.csv', sep=';', header=0)

features, label = CreateDataset(100, df)
label = np.array(label)
label = np.reshape(label, (label.size, 1))  # reshape the array from [label] (3540,) to [samples, label] (3540,1)

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# zero-offset class values
y_train = y_train - 1
y_test = y_test - 1
# one hot encode y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(LSTM(100, input_shape=(100, 9)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=4)
model.evaluate(X_test, y_test, batch_size=4)

"""

df = pd.read_csv('AllData.csv', sep=';')
label = df.pop('Action').replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 0)
label = label.replace(12, 1)
features = np.array(df)

normalize = preprocessing.Normalization()
normalize.adapt(features)

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

c = SVC(gamma=2, C=1)
c.fit(X_train, y_train)
score = c.score(X_test, y_test)
y_pred = c.predict(X_test)
print(confusion_matrix(y_test, y_pred))
"""