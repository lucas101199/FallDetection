import keras as keras
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('AllData.csv', sep=';', header=0)
df.pop('Sensor_ID')
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_label = np.array(train_df.pop('Action'))
train_features = np.array(train_df)

test_label = np.array(test_df.pop('Action'))
test_features = np.array(test_df)

scaler = MinMaxScaler(feature_range=(0, 5))
scaled = scaler.fit_transform(train_features)
scaled = pd.DataFrame(scaled)

model = tf.keras.Sequential([
    keras.layers.Embedding(input_dim=train_features.shape[-1], output_dim=256),
    keras.layers.GRU(256, return_sequences=True),
    keras.layers.SimpleRNN(128),
    keras.layers.Dense(12)
])

model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

model.fit(scaled, train_label, epochs=2, batch_size=64)

print(model.output_shape)
