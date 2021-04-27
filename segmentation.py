# sliding windows of 2 sec and 50% overlap
# time pass in argument is in second and return a set of windows
import pandas as pd


def WriteData(data, label):
    file = open('raw_data.txt', 'a+')
    line = ''
    for i in range(len(data)):
        for j in range(3):
            line += str(data[i][j]) + ' '
    line += label[0] + '\n'
    file.write(line)


def sliding_window(file, time):
    df = pd.read_csv(file)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], infer_datetime_format=True)
    timestamp = df['TimeStamp'].values
    set_windows = []
    start_time = timestamp[0]
    end_time = timestamp[-1]
    while start_time + pd.offsets.Second(time) <= end_time + pd.offsets.Second(1):
        data = df.loc[
            (df['TimeStamp'] >= start_time) & (df['TimeStamp'] <= start_time + pd.offsets.Second(time))].values
        label = data[:, 4]
        data = data[:, 1:4]
        WriteData(data, label)
        start_time += pd.offsets.Second(time / 2)
        set_windows.append(data)
    return set_windows


# 192.168.0.41
