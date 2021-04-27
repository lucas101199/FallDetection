def getSensors():
    sensorList = ['Accelerometer: x-axis',
           'Accelerometer: y-axis',
           'Accelerometer: z-axis',
           'Velocity: value'
          ]
    return sensorList

def getFeatures():
    ftList = ['Mean',
        'StandardDeviation',
        'RootMeanSquare',
        'MaximalAmplitude',
        'MinimalAmplitude',
        'Median',
        'Number of zero-crossing',
        'Skewness',
        'Kurtosis',
        'First Quartile',
        'Third Quartile',
        'Autocorrelation',
        'Energy'
        ] 
    return ftList