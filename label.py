import sys
import re
from datetime import datetime

import pandas as pd
import numpy as np


def ParseAnnotationsFile(file):
    # dict annotations  with format {1: [start, end, label]}
    annotations = {}
    df = pd.read_csv(file)['label']
    line = re.sub('[{}"[]', '', df.values[0])  # delete {}]" characters
    line = line.split(']')
    for i in range(len(line)):
        if line[i] != '':
            value = line[i].split(",")
            for j in value:
                if j == '':
                    value.remove(j)
            val_dict = [value[0].split(' ')[1], value[1].split(' ')[2], value[3].split(' ')[2]]
            annotations[i] = val_dict
    return annotations


# write the new column with the label in the csv file and delete the data with no label
def WriteLabelInFile(dict, file):
    df = pd.read_csv(file)
    df['Class'] = np.nan
    for k, v in dict.items():
        start = float(v[0])
        end = float(v[1])
        label = v[2]
        index_time_start = df.loc[df['TimeStamp'] >= start].head(1).index.values.astype(int)[0]
        index_time_end = df.loc[df['TimeStamp'] >= end].head(1).index.values.astype(int)[0]
        df.at[index_time_start:index_time_end, 'Class'] = label

    df = df[df.Class.notnull()]
    df = df.round(decimals=3)
    df.to_csv(file, index=False)


file_name = sys.argv[1]
label_name = sys.argv[2]

annotations = ParseAnnotationsFile(label_name)
WriteLabelInFile(annotations, file_name)
