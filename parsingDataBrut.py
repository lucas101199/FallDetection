# Need to add a line break at the end of the file.txt if it doesn't have one
import sys


file_name = sys.argv[1]
file_write = sys.argv[2]

file = open(file_name, 'r')
fileToWrite = open(file_write, 'w')
lines = file.readlines()
features = 'TimeStamp,X-Axis,Y-Axis,Z-Axis\n'
fileToWrite.write(features)
array_reversed = []
for line in lines:
    line_split = line.split(';')
    linetowrite = line_split[3][:-1] + ',' + line_split[0] + ',' + line_split[1] + ',' + line_split[2] + '\n'
    array_reversed.append(linetowrite)

fileToWrite.writelines(list(reversed(array_reversed)))
