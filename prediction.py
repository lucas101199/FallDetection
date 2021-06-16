import sys
import numpy as np
from tensorflow import keras


text = sys.argv[1]
text = text.replace('[', '')
text = text.replace(']', '')
data = np.fromstring(text, dtype=float, sep=',')
data = data.reshape(1, 200, 3)
model = keras.models.load_model("/home/gael/FallDetection/rnn.h5")
print(model.predict_classes(data)[0][0])
# print(model.predict(data))  # if we want the probability instead of the classe : probability [0,1]
