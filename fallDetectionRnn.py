# Les import necessaires.
from os import listdir
from os.path import isfile, join
import pandas as pd
import keras as keras
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import StandardScaler


def GetLabelFromFile(f, labels):
    for k, v in labels.items():
        if f.find(v) != -1:
            return k
    return -1

# Récuperer tous les fichiers qu'il y'a dans le dossier "UMAFall_Dataset".
allfiles = [f for f in listdir("UMAFall_Dataset") if isfile(join("UMAFall_Dataset", f))]

# Préparer les features.
features = "TimeStamp;Sample No;X-Axis;Y-Axis;Z-Axis;Sensor_Type;Sensor_ID\n"

# pour chaque fichier qu'il y a dans le dossier "UMAFall_Dataset", lire ce fichier,
# et enregister son contenue dans un new_file,
# le but est de concaténer le contenue de tous les fichier qui se trouvent dans "UMAFall_Dataset".
for f in allfiles:
    file = open(join("UMAFall_Dataset", f), "r")
    lines = file.readlines()
    file.close()
    if lines[0] != features:
        output = [features]
        new_file = open(join("UMAFall_Dataset", f), "w+")
        for i in range(len(lines)):
            if lines[i].find("%") == 0 or len(lines[i].split()) == 0:
                pass
            else:
                output.append(lines[i])
        new_file.writelines(output)
        new_file.close()

# Préparation des différents Labels possible allant de 'Aplausing' à 'Fall'

labels = {0: "Aplausing", 1: "HandsUp", 2: "MakingACall", 3: "OpeningDoor",
          4: "Sitting_GettingUpOnAChair", 5: "Walking", 6: "Bending", 7: "Hopping",
          8: "Jogging", 9: "LyingDown", 10: "GoDownstairs", 11: "GoUpstairs", 12: "Fall"}

features_action = features[:-1] + ';Action\n'
allData = open('AllData.csv', 'w')
allData.write(features_action)
allData.close()

# Enregistrer le contenu de tous les fichier qui se troyvent dans "UMAFall_Dataset" dans un seul dataset très grand nomé 'AllData.csv'

for f in allfiles:
    name_file = join("UMAFall_Dataset", f)
    label = GetLabelFromFile(name_file, labels)
    with open(name_file) as csv_file:
        df = pd.read_csv(name_file, sep=';', header=0)
        df = df[df.Sensor_ID == 3]
        action = [label] * len(df.index)
        df["Action"] = action
        df.to_csv(r'AllData.csv', mode='a', header=False, index=False, sep=';')

# Lire le dataset 'AllData.csv'

Label = {0 : 'fallen', 1 : 'not_fallen'}

data_not = pd.read_csv('AllData.csv', sep=';', header=0)

"""
# Préparation des targets pour ce dataset.

labels = []
for i in range(len(data_not)):
    if (i % 2) == 0 :
        labels.append(1)
    else :
        labels.append(0)

labels = np.array(labels)

labels.shape
"""
# create the labels from the column Action
labels = data_not.pop('Action')
features = data_not.values # get the features with the remaining columns



# Standarisation des input.
scaler = StandardScaler()
features = scaler.fit_transform(features)

features.mean(axis = 0)

features.std(axis = 0)

features3d = features.reshape(features.shape[0] , 1 , 7)

# L'architecture de notre model.
model = tf.keras.Sequential([
    keras.layers.GRU(256, return_sequences=True , input_dim = 7),
    keras.layers.SimpleRNN(128),
    keras.layers.Dense(1)
])

model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

model.fit(features3d, labels, epochs=10, batch_size=64)

"""**Commen on peut le voir, on a un MSE de 0.2503, afin d'augmenter les performances de notre système on peut:**

*   Complixifier l'architecture de notre réseau de neuronnes.
*   Augmenter le nombre d'épochs.




"""
