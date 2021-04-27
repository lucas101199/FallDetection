# sliding windows of 2 sec and 50% overlap
# time pass in argument is in second and return a set of windows
import pandas as pd


def sliding_window(file, time):
    df = pd.read_csv(file)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], infer_datetime_format=True)
    timestamp = df['TimeStamp'].values
    set_windows = []
    start_time = timestamp[0]
    end_time = timestamp[-1]
    while start_time + pd.offsets.Second(time) <= end_time + pd.offsets.Second(1):
        data = df.loc[(df['TimeStamp'] >= start_time) & (df['TimeStamp'] <= start_time + pd.offsets.Second(time))]
        start_time += pd.offsets.Second(time / 2)
        set_windows.append(data)
    if len(set_windows) == 0:
        set_windows = df
    return set_windows


# 192.168.0.41
