import csv

header = ['meanX', 'meanY', 'meanZ', 'maxX', 'maxY', 'maxZ', 'minX', 'minY', 'minZ',
          'stdX', 'stdY', 'stdZ', 'varX', 'varY', 'varZ', 'medianX', 'medianY',
          'medianZ', 'rmsX', 'rmsY', 'rmsZ', 'corX', 'corY', 'corZ', 'energyX',
          'energyY', 'energyZ', 'iqrX', 'iqrY', 'iqrZ', 'madX', 'madY', 'madZ', 'sma', 'Class']

row_list = [header]
file = open('features_normt.txt', 'r')
labelfile = open('labelt.txt', 'r')
lines_feat = file.readlines()
lines_label = labelfile.readlines()
for i in range(len(lines_feat)):
    array_feat = []
    for feat in lines_feat[i].split(' '):
        if feat[-1] == '\n':
            array_feat.append(feat[:-1])
        else:
            array_feat.append(feat)
    array_feat.append(lines_label[i][:-1])
    row_list.append(array_feat)
with open('features_normt.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)
