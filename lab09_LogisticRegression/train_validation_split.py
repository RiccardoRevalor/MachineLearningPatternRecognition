import numpy as np


def splitTrainingValidation(percentageTraining, D, L, seed=0):
    """"
    Function to splot the data into 2 partitions: the training one and the validation one.
    The splitting is done randomly so the seed needs to be initialized first.

    Params:
    - percentageTraining: percentage of training data over the overall data 
    - D
    - L
    - seed: valie of the seed for the random split, default is 0

    Returned Values:
    - (DTR, LTR), (DVAL, LVAL):
        DTR and LTR are model training data and labels
        DVAL and LVAL are validation data and labels
    """

    #set cardinality of training partition
    #Obviously since we cannot have fractional samples, this number is rounded
    nTrain = int(D.shape[1] * percentageTraining)

    #initialize the seed for the random split
    np.random.seed(seed)

    #shuffle the data indices by making a random permutation
    """
    example of usage: 
    np.random.permutation(100) returns: 
    array([45, 97, 37, 87, 36,  7,  0, 23, 24, 31, 75, 35, 68, 53, 65,  8, 84,
       41, 20, 56, 81, 12, 47, 72, 63, 38,  6, 58, 91, 48, 92, 88, 71, 29,
       46, 10, 14, 18, 27, 98, 67, 76, 78, 54, 90, 17, 50, 55,  4, 21, 19,
       32, 95, 15,  5, 64, 93, 69, 52, 89,  2, 42, 43, 82, 44, 99, 85, 13,
       34, 94, 74, 80, 70,  3, 77, 73,  1, 11, 28, 16, 62, 96, 22, 66, 49,
       57, 60, 51,  9, 59, 26, 33, 39, 61, 79, 86, 40, 83, 25, 30],
      dtype=int32)
    """

    shuffledIndices = np.random.permutation(D.shape[1])

    #Split randomically accordin to the percentage
    #TRAINING DATA
    DTR = D[:, shuffledIndices[0:nTrain]]
    LTR = L[shuffledIndices[0:nTrain]]

    #VALIDATION DATA
    DVAL = D[:, shuffledIndices[nTrain:]]
    LVAL = L[shuffledIndices[nTrain:]]

    return (DTR, LTR), (DVAL, LVAL)


