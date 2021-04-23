import sys


file_name = sys.argv[1]
file_write = sys.argv[2]

file = open(file_name, 'r')
fileToWrite = open(file_write, 'w')
lines = file.readlines()
features = 'TimeStamp,X-Axis,Y-Axis,Z-Axis,Velocity\n'
fileToWrite.write(features)
lines_split_space = lines[0].split(' ')
firstLine = lines_split_space[1] + ','
fileToWrite.write(firstLine)
del lines_split_space[0]
del lines_split_space[0]
for i in range(len(lines_split_space)):
    line_split = lines_split_space[i].split(";")
    if len(line_split) == 1:
        fileToWrite.write(line_split[0] + ',')
    else:
        fet = line_split[:-1]
        allfeat = ''
        for data in fet:
            data = data.replace(',', '.')
            allfeat += data + ','
        fileToWrite.write(allfeat[:-1] + '\n')