#CONFUSION MATRIX COMPUTATIONS

import numpy as np

#import MVG model 
#IMPOORTANT: adjust the path to the MVG model based on your system
import sys
MVG_path = './models_finished/MVG'
if not MVG_path in sys.path:
    sys.path.append(MVG_path)

import MVG


def computeConfMatrix(PVAL, LVAL):
    """
    Compute the confusion matrix for the predicted labels and the actual labels.
    Args:
    - PVAL: Predicted labels
    - LVAL: Actual labels
    Returns:
    - Confusion matrix
    """
    numClasses = np.unique(LVAL).shape[0] #number of classes
    ConfMatrix = np.zeros((numClasses, numClasses)) #initialize the confusion matrix with zeros

    for classPredicted in range(numClasses):
        #for each class find the tre positives and ALL the false negatives

        classRow = np.array([]) #initialize the classRow with an empty array

        for classActual in range(numClasses):
            if classActual == classPredicted: 
                TP = np.sum((PVAL == classPredicted) & (LVAL == classPredicted))
                classRow = np.append(classRow, TP)
                continue

            #compute each FP for each wrongly assigned label
            FPi = np.sum((PVAL == classPredicted) & (LVAL == classActual))

            #add FPi to the classCol
            classRow = np.append(classRow, FPi)

        
        #add classCol to the confusion matrix
        ConfMatrix[classPredicted, :] = classRow


    return ConfMatrix



#always import MVG before using this function!

def computeConfMatrixFromLL(LVAL, logLikelihoods, Priors, useLog=True):
    """
    Compute the confusion matrix for the predicted labels and the actual labels.
    Args:
    - LVAL: Actual labels
    - logLikelihoods: matriix of log likelihoods for each class
    - Priors: array of priors for each class, priors are application dependent
    - useLog: if True, use log likelihoods, else use normal likelihoods

    Returns:
    - Confusion matrix
    """

    SJoint = MVG.computeSJoint(logLikelihoods, Priors, useLog=useLog) #compute the joint densities by multiplying the score matrix S with the Priors
    SPost = MVG.computePosteriors(SJoint, useLog=True)  #compute the posteriors by normalizing the joint densities
    PVAL = np.argmax(SPost, axis=0) #select the class with the highest posterior probability for each sample, set axis=0 to select the class with the highest posterior probability for each sample

    #call the computeConfMatrix function to compute the confusion matrix
    return computeConfMatrix(PVAL, LVAL)
    
