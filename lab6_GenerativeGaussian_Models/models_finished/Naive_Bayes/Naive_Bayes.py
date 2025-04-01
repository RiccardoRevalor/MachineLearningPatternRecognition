#NAIVE BAYES MODEL

import numpy as np
from mean_covariance import vcol, vrow, compute_mu_C                #for vcol, vrow, compute_mu_C functions
from scipy.special import logsumexp                                 #for scipy.special.logsumexp
from logpdf_loglikelihood_GAU import logpdf_GAU_ND                  #for computing the log-likelihood of the Gaussian distribution



def computeParams_ML_NaiveBayesAssumption(D, labels):
    #Compute the ML (Maximum Likelihood) parameters of the MVG distribution given the data and the labels, and use the Naive Bayes assumption (so Covariance Matrices are diagonal)
    """
    Parameters:
    - D: the data matrix of shape (numFeatures, numSamples)
    - labels: the labels of the data, so a list of length numSamples

    Returned Values:
    - params: the model parameters, so  list of tuples (mu, C) where mu is the mean vector fo class c and C is diagonal the covariance matrix of class c
    """

    params = []
    numClasses = np.unique(labels).shape[0] #number of classes
    for label in range(numClasses):
        #compute MLE estimates of mean and covariance matrix for each class i
        mu, C = compute_mu_C(D[:, labels == label])
        params.append((mu, np.diag(np.diag(C)))) #append the mean vector and the diagonal covariance matrix to the list of parameters

    return params #params is a list of tuples (mu, C) where mu is the mean vector fo class c and C is the diagonal covariance matrix of class c






def scoreMatrix_Pdf_GAU(D, params, useLog=False):  
    #Compute the (log?)-Pdf of the data given the parameters of a Gaussian distribution and populate the score matrix S with the (log?)-pdf of each class
    """
    Parameters:
    - D: the data matrix of shape (numFeatures, numSamples)
    - params: the model parameters, so  list of tuples (mu, C) where mu is the mean vector fo class c and C is the covariance matrix of class c
    - useLog: if True, compute the log-pdf, else compute the pdf

    Returned Values:
    - S: the score matrix of shape (numClasses, numSamples) where each row is the score of the class given the sample

    """

    #The score matrix is filled with the pdfs of the training data given the MLE parameters of the MVG distribution
    #S[i, j] is the pdf of the j-th sample given the i-th class
    
    numClasses = len(params) #number of classes, since for each class we have a tuple (mu, C)
    S = np.zeros((numClasses, D.shape[1]))
    for label in range(numClasses):
        if useLog:
            #if useLog is True, then compute the log-pdf
            S[label, :] = logpdf_GAU_ND(D, params[label][0], params[label][1])
        else:
            #if useLog is False, then compute the pdf
            S[label, :] = np.exp(logpdf_GAU_ND(D, params[label][0], params[label][1]))

    return S



def computeSJoint(S, Priors, useLog=False):
    # Compute the joint densities by multiplying the score matrix S with the Priors
    """
    Parameters:
    - S: the score matrix of shape (numClasses, numSamples) where each row is the score of the class given the sample
    - Priors: the priors of the classes, so a list of length numClasses
    - useLog: if True, compute the log-joint densities, else compute the joint densities

    Returned Values:
    - SJoint: the (log?)joint densities of shape (numClasses, numSamples) where each row is the joint density of the class given the sample
    """
    

    if (useLog):
        #S needs to be already in log scale, so we just need to add the log of the priors
        return S + vcol(np.log(Priors)) #multiply each row of S (where 1 row corresponds to a class) with the prior of the class
    else:
        return S * vcol(Priors)
    


def computePosteriors(SJoint, useLog=False):
    """
    Compute the posteriors by normalizing the joint densities
    The posteriors are the joint densities divided by the sum of the joint densities which are the marginals

    Parameters:
    - SJoint: the joint densities of shape (numClasses, numSamples) where each row is the joint density of the class 

    Returned Values:
    - SPost: the posteriors of shape (numClasses, numSamples) where each row is the posterior of the class given the sample
    """
    if useLog:
        #1. Compute marginals usign the logsumexp trick to minimize numerical problems
        #logsumexp is a function that computes the log of the sum of exponentials of input elements
        #It is more numerically stable than computing the sum of exponentials directly
        #It computes log(exp(a) + exp(b)) in a numerically stable way

        #sum over the rows (axis=0) to get the marginal of each sample
        SMarginal = logsumexp(SJoint, axis=0)
        #SMarginal has now shape = (numSamples, ) -> it's a row vector
        #I need to make it of shape (1, numSamples) 
        SPost = SJoint - vrow(SMarginal) #element wise division in log scale, so I just need to subtract the marginals from the joint densities
        

    else:
        
        #1. Compute marginals
        SMarginal = vrow(SJoint.sum(0)) #sum over the rows (axis=0) to get the marginal of each sample

        #2. Compute posteriors by dividing the joint densities by the marginals
        SPost = SJoint / SMarginal #element wise division

    return SPost
   