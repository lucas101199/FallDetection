import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal


def median_filter(data, f_size):
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    for i in range(num_signal):
        f_data[:, i] = signal.medfilt(data[:, i], f_size)
    return f_data


def plot_lines(data, fs, title):
    num_rows, num_cols = data.shape
    if num_cols != 3:
        raise ValueError('Not 3D data')
    fig, ax = plt.subplots()
    labels = ['x', 'y', 'z']
    color_map = ['r', 'g', 'b']
    index = np.arange(num_rows) / fs
    for i in range(num_cols):
        ax.plot(index, data[:, i], color_map[i], label=labels[i])
    ax.set_xlim([0, num_rows / fs])
    ax.set_xlabel('Time [sec]')
    ax.set_title('Time domain: ' + title)
    ax.legend()


def plot3D(data, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], zdir='z')
    ax.set_title(title)


# change the original data with the new filtered data
# return a dataframe with the filtered values instead of the original
def ChangeData(file, kernel_size):
    df = pd.read_csv(file)
    x = df.drop(['TimeStamp'], axis=1).values
    timestamp = df.loc[:, 'TimeStamp'].values
    label = df.loc[:, 'Class'].values
    median_data = median_filter(x, kernel_size)
    new_df_dict = {'TimeStamp': timestamp, 'X-Axis': median_data[:, 0], 'Y-Axis': median_data[:, 1],
                   'Z-Axis': median_data[:, 2], 'Class': label}
    return pd.DataFrame(new_df_dict)


df = pd.read_csv('balais_4_L.csv')
data = df.drop(['TimeStamp'], axis=1).values
plot_lines(data, 50, 'balais')
plt.show()
