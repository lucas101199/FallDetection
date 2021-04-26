import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('An error ocurred while creating direcory: "' +  directory +'"')
