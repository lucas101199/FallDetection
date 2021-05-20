import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

"""
# Create a dataset from a DataFrame of size [samples, timesteps, features] and size [samples, action] for the label
def CreateDataset(TimeSteps, df):
    size = math.ceil(len(df) / TimeSteps)
        features = np.empty([size, TimeSteps, 9])
        j = 0
    label = []
    for i in range(0, len(df), TimeSteps):
        features[j] = df.loc[i:i + TimeSteps - 1,
                      ['X-AxisA', 'Y-AxisA', 'Z-AxisA', 'X-AxisG', 'Y-AxisG', 'Z-AxisG',
                       'X-AxisM', 'Y-AxisM', 'Z-AxisM']].values
        action = df.loc[i, ['Class']].values
        label.append(action)
        j += 1
    return features, label
"""


def extractLabel(lines):
    label = []
    for line in lines:
        label.append(int(line[-2]))
    return label


def extractData(lines):
    data = []
    for line in lines:
        array = []
        xyz = []
        linesplit = line.split(' ')
        for i in range(len(linesplit)):
            if len(xyz) == 3:
                array.append(xyz)
                xyz = []
            xyz.append(float(linesplit[i]))
        data.append(array)
    return data


df = open('raw_data.txt', 'r')
lines = df.readlines()
lines = np.array(lines)
label = np.array(extractLabel(lines)).reshape(len(lines), 1)
data = np.array(extractData(lines))
pos = 0
neg = 0

for i in range(label.shape[0]):
    if label[i][0] == 0:
        neg += 1
    else:
        pos += 1

    # label = np.reshape(label, (label.size, 1))  # reshape the array from [label] (3540,) to [samples, label] (3540,1)
weight0 = (1 / neg) * (neg + pos) / 2.0
weight1 = (1 / pos) * (neg + pos) / 2.0
class_weight = {0: weight0, 1: (weight1 / 4)}

# Encode the label
# le = preprocessing.LabelEncoder()
# label = le.fit_transform(label)

# Split the dataset between train and test
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)

# show the size of each cut
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
"""
# zero-offset class values
y_train = y_train - 1
y_test = y_test - 1
# one hot encode y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
"""
initial_bias = -10  #np.log([pos/neg])
initial_bias = tf.keras.initializers.Constant(initial_bias)
print(initial_bias.value)
# le model de la machine ss
model = Sequential()
model.add(LSTM(100, input_shape=(200, 3), return_sequences=True))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid', bias_initializer=initial_bias))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=20, batch_size=40)
# modification of fit model
history = model.fit(X_train, y_train, epochs=20, batch_size=4, class_weight=class_weight)
model.save('saved_model/mlp')
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/mlp')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# model.fit(X_train, y_train, epochs=20, batch_size=4)

model.evaluate(X_test, y_test, batch_size=2)
ypred = model.predict_classes(X_test, batch_size=2)
yp = np.argmax(ypred, axis=1)
print(confusion_matrix(y_test, ypred))
# list all data in history
# print(history.history.keys())
"""
# plot the training and validation accuracy and loss at each epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("accuracy.png")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot.savefig("loss.png")
"""
