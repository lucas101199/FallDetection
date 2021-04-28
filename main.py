from segmentation import sliding_window
from preprocess import ChangeData
from features import WriteFeatures
from sklearn.preprocessing import MinMaxScaler


def normalizeData(file):
    f = open(file, 'r')
    lines = f.readlines()
    data_normalize = []
    for line in lines:
        features = []
        for l in line.split(' '):
            features.append(int(l))
        data_normalize.append(features)
    scaler = MinMaxScaler()
    return scaler.fit_transform(data_normalize)


# for 1 file
file = 'test.csv'
filetxt = 'raw_data.txt'
#df = ChangeData(file, 3)  # preprocess
#sliding_window(df, 2)  # segmentation
WriteFeatures(filetxt)  # features extraction
