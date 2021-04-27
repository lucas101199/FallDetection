import numpy as np
import pandas as pd
import statistics as st
from scipy.stats import kurtosis, skew, entropy, iqr, median_abs_deviation


# Sum of the squares divided by the number of values
def energy(val):
    return st.mean([val[i]**2 for i in range(len(val))])


def sma(x, y, z):
    sum = 0
    for i in range(len(x)):
        sum += (abs(x[i]) + abs(y[i]) + abs(z[i]))
    return sum / len(x)


def GetFeatures(x, y, z):
    xrms = 0
    yrms = 0
    zrms = 0
    for i in range(len(x)):
        xrms += x[i]**2
        yrms += y[i]**2
        zrms += z[i]**2
    xcor = np.correlate(x, y, mode='full')
    ycor = np.correlate(y, z, mode='full')
    zcor = np.correlate(z, x, mode='full')
    features = ''
    features += str(st.mean(x)) + ' '
    features += str(st.mean(y)) + ' '
    features += str(st.mean(z)) + ' '
    features += str(max(x)) + ' '
    features += str(max(y)) + ' '
    features += str(max(z)) + ' '
    features += str(min(x)) + ' '
    features += str(min(y)) + ' '
    features += str(min(z)) + ' '
    features += str(st.stdev(x)) + ' '
    features += str(st.stdev(y)) + ' '
    features += str(st.stdev(z)) + ' '
    features += str(np.var(np.array(x))) + ' '
    features += str(np.var(np.array(y))) + ' '
    features += str(np.var(np.array(z))) + ' '
    features += str(np.median(np.array(x))) + ' '
    features += str(np.median(np.array(y))) + ' '
    features += str(np.median(np.array(z))) + ' '
    features += str((np.sqrt(xrms) / np.sqrt(len(x)))) + ' '
    features += str((np.sqrt(yrms) / np.sqrt(len(y)))) + ' '
    features += str((np.sqrt(zrms) / np.sqrt(len(z)))) + ' '
    features += str(np.median(xcor))
    features += str(np.median(ycor))
    features += str(np.median(zcor))
    features += str(entropy(x)) + ' '
    features += str(entropy(y)) + ' '
    features += str(entropy(z)) + ' '
    features += str(energy(x)) + ' '
    features += str(energy(y)) + ' '
    features += str(energy(z)) + ' '
    features += str(iqr(x)) + ' '
    features += str(iqr(y)) + ' '
    features += str(iqr(z)) + ' '
    features += str(median_abs_deviation(x)) + ' '
    features += str(median_abs_deviation(y)) + ' '
    features += str(median_abs_deviation(z)) + ' '
    features += str(sma(x, y, z)) + '\n'
    return features


file = open('raw_data.txt', 'r')
feature_file = open('features.txt', 'a+')
label_file = open('label.txt', 'a+')
lines = file.readlines()
for line in lines:
    line_split = line.split(' ')
    label = line_split[-1]
    line_split = line_split[:-1]
    x = list(map(int, line_split[::3]))
    y = list(map(int, line_split[1::3]))
    z = list(map(int, line_split[2::3]))
    feature_file.write(GetFeatures(x, y, z))
    label_file.write(label + ' ')

feature_file.close()
label_file.close()
file.close()


