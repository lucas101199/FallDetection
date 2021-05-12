"""file = open('NotFall1.txt', 'r')
fileToWrite = open('not.csv', 'w')
lines = file.readlines()
features = 'TimeStamp,X-Axis,Y-Axis,Z-Axis\n'
fileToWrite.write(features)

for line in lines:
    halfLine1 = line[:6]
    halfline2 = line[7:-1]
    fileToWrite.write(halfline2 +";"+ halfLine1)
    newLine = halfline2 +";"+ halfLine1
  
file.close()
fileToWrite.close()

    

    """