# sliding windows of 2 sec and 50% overlap
# time pass in argument is in second and return a set of windows
# frequence 100Hz
import pandas as pd

FREQUENCY = 100


def WriteData(data, label):
    file = open('raw_data.txt', 'a+')
    line = ''
    for i in range(len(data)):
        for j in range(3):
            line += str(data[i][j]) + ' '
    line += str(int(label[0])) + '\n'
    file.write(line)


def sliding_window(df, time):
    size_window = time * FREQUENCY
    for i in range(0, len(df), int(size_window/2)):
        data = df[i:(i+size_window)].values
        if len(data) == size_window:
            label = data[:, 4]
            data = data[:, 1:4]
            WriteData(data, label)


# 192.168.0.41
