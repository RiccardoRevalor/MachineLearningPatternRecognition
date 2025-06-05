import numpy as np

def loadDataSet(path, numFeatures):
    rawData = np.genfromtxt(path, delimiter=',', dtype = "str")

    #extract the features for each footprint
    features = rawData[:, 0:numFeatures]
    features = np.array(features, dtype= np.float64).T

    #extract the labels
    classLabels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2} 
    labels = np.array([classLabels[name] for name in rawData[:, 4]], dtype=int)

    return features, labels

