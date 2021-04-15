import math
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Bidirectional
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split


# Create a dataset from a DataFrame of size [samples, timesteps, features] and size [samples, action] for the label
def CreateDataset(TimeSteps, df):
    size = math.ceil(len(df) / TimeSteps)
    features = np.empty([size, TimeSteps, 6])
    j = 0
    label = []
    for i in range(0, len(df), TimeSteps):
        features[j] = df.loc[i:i + TimeSteps - 1,
                      ['X-AxisA', 'Y-AxisA', 'Z-AxisA', 'X-AxisG', 'Y-AxisG', 'Z-AxisG']].values
        action = df.loc[i:i + TimeSteps - 1, ['Action']].values
        action = action.flatten()
        label.append(np.bincount(action).argmax())
        j += 1
    return features, label


df = pd.read_csv('AllData.csv', sep=';', header=0)

features, label = CreateDataset(60, df)
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
model.add(LSTM(64, input_shape=(60, 6), return_sequences=True))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=4)

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
model.save('saved_model/model.h5')


