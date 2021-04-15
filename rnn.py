import math
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Bidirectional
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split


def CreateDataset(TimeSteps, df):
    size = math.ceil(len(df) / TimeSteps)
    array_axis = np.empty([size, TimeSteps, 6])
    j = 0
    label = []
    for i in range(0, len(df), TimeSteps):
        array_axis[j] = df.loc[i:i + TimeSteps - 1,
                        ['X-AxisA', 'Y-AxisA', 'Z-AxisA', 'X-AxisG', 'Y-AxisG', 'Z-AxisG']].values
        action = df.loc[i:i + TimeSteps - 1, ['Action']].values
        action = action.flatten()
        label.append(np.bincount(action).argmax())
        j += 1
    return array_axis, label


df = pd.read_csv('AllData.csv', sep=';', header=0)

array, label = CreateDataset(60, df)
label = np.array(label)
label = np.reshape(label, (label.size, 1))

X_train, X_test, y_train, y_test = train_test_split(array, label, test_size=0.3, random_state=42)

# zero-offset class values
y_train = y_train - 1
y_test = y_test - 1
# one hot encode y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(LSTM(100, input_shape=(60, 6), return_sequences=True))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=30)

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

