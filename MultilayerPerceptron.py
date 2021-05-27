import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

feature_vector_length = 34
num_classes = 2

df = pd.read_csv('features_norm.csv')
Y = df['Class']
X = df.drop(['Class'], axis=1)
X = np.array(X)
Y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(350, input_shape=(feature_vector_length,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=4)

model.save('saved_model/mlp')
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/mlp')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
