import numpy as np
from logpdf_loglikelihood_GAU import logpdf_GAU_ND #for single density of a gaussian component
from scipy.special import logsumexp #for marginalizing the joints to retrieve the GMM log density
from mean_covariance import vrow, vcol, compute_mu_C #for computing the mean and covariance of a Gaussian component


def logpdf_GMM(X, gmm):
    """
    Compute log density of data points X under a Gaussian Mixture Model (GMM).
    Parameters
    - X: matrix of size (D, N) where D is the number of features and N is the number of data points.
    - gmm: list of gaussian components. Each one is a tuple of (weight, mean, covariance).
           weight: scalar
           mean: vector of size (D,)
           covariance: matrix of size (D, D)
        Example: gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]
    Returns
    - logpdf: vector of size (N,) containing the log density of each data point under the GMM.
    """

    #1. create matrix S of shape (K, N), where N = number of samples and K = number of components
    K = len(gmm)  # number of components
    N = X.shape[1]
    S = np.zeros((K, N))

    #iterate over components, for each componente take mean, covariance and compute log density of the Gaussian
    for k in range(K):
        weight, mean, covariance = gmm[k]
        S[k, :] = logpdf_GAU_ND(X, mean, covariance) + np.log(weight)
    
    #these are the log joints, then GMM can be casted as a latent variable model, i.e. we can find the GMM log density marginalizing the joints over the latent variable
    #latent variable if the component/cluster
    logdens = logsumexp(S, axis=0)  #marginalize joints over the components

    return logdens  






################################################################### GMM PIPELINE #####################################################################################################


def smooth_Covariance_Matrix(c, psiEig):
    U, s, _ = np.linalg.svd(c)
    s[s< psiEig] = psiEig
    return np.dot(U, vcol(s)*U.T)

def GMM_EM_iteration(X, gmm_start, psiEig=None):
   """
      One single iteration of the EM algorithm for Gaussian Mixture Models (GMM).
      Parameters
      -X: matrix of size (D, N) where D is the number of features and N is the number of data points.
      -gmm_start: list of starter GMM components. Can be obtained with either K-Means or LGB Algorithm.
               Each one is a tuple of (weight, mean, covariance).
               weight: scalar
               mean: vector of size (D,)
               covariance: matrix of size (D, D)
      -threshold_stop: threshold for stopping the EM algorithm. If the change in log likelihood is less than this value, stop.
      Returns
      -gmm: list of gaussian components. Each one is a tuple of (weight, mean, covariance).
            weight: scalar
            mean: vector of size (D,)
            covariance: matrix of size (D, D)
    """
    
   #1. E-STEP: compute responsibilities
   #create matrix S of shape (K, N), where N = number of samples and K = number of components
   K = len(gmm_start)  # number of components
   N = X.shape[1]
   S = np.zeros((K, N))

   #iterate over components, for each componente take mean, covariance and compute log density of the Gaussian
   for k in range(K):
      weight, mean, covariance = gmm_start[k]
      S[k, :] = logpdf_GAU_ND(X, mean, covariance) + np.log(weight)


   #for each sample, marginalize the log-joints over the components to get the log-marginal
   logdens = logsumexp(S, axis=0)  #vector of size (N,)

   #compute log-posteriors by removing log-marginal from the log-joints
   log_posteriors = S - logdens   #(K, N) - (N,) -> broadcasting -> (K, N) - (N,N) = (K, N) thanks to broadcasting

   #compute responsibilities, so cluster posteriors, by exponentiating the log-posteriors
   responsibilities = np.exp(log_posteriors) #(K, N)


   #2. M-STEP: estimate new GMM parameters
   gmm = []
   #compute zero, first, and second order statistics from the responsibilities of each cluster
   #for each cluster k, do:
   for k in range(K):
      gamma = responsibilities[k, :]
      Z_gamma = np.sum(gamma) #zero order
      F_gamma = vcol(np.sum(vrow(gamma) * X, axis = 1)) #first order
      S_gamma = (vrow(gamma) * X) @ X.T #second order, (D, N) @ (N, D) = (D, D)

      #ESTIMATE NEW PARAMS for the cluster k
      mu_k_new = F_gamma / Z_gamma  #col vector (D, 1)
      cov_k_new = S_gamma / Z_gamma - vcol(mu_k_new) @ vrow(mu_k_new)  #covariance matrix, (D, D)
      #if psiEig is provided, we have to contrain te eigvalues of the covariance matrix
      if psiEig is not None: cov_k_new = smooth_Covariance_Matrix(cov_k_new, psiEig)
      weight_k_new = Z_gamma / N #n = sum of reponsibilities for each sample for each cluster k = total number of samples, since responsibilities of each sample sum to 1 being fractions
      gmm.append((weight_k_new, mu_k_new, cov_k_new))  #append new params for the cluster k

   return gmm  #return the new GMM parameters after one EM iteration



def train_GMM_EM(X, gmm_start, threshold_stop=1e-6, psiEig=None, max_iter=1000):
   """
   EM algorithm for Gaussian Mixture Models (GMM). 
   Parameters
   - X: matrix of size (D, N) where D is the number of features and N is the number of data points.
   - gmm_start: list of starter GMM components. Can be obtained with either K-Means or LGB Algorithm.
            Each one is a tuple of (weight, mean, covariance).
            weight: scalar
            mean: vector of size (D,)
            covariance: matrix of size (D, D)
   - threshold_stop: threshold for stopping the EM algorithm. If the change in log likelihood is less than this value, stop.
   - max_iter: maximum number of iterations for the EM algorithm.
   Returns
   - gmm: list of gaussian components. Each one is a tuple of (weight, mean, covariance).
            weight: scalar
            mean: vector of size (D,)
            covariance: matrix of size (D, D)
   """

   gmm_old = gmm_start.copy()  
   num_iters = 0
   while True:
      #compute the log likelihood with old GMM params
      GMM_ll_old = logpdf_GMM(X, gmm_old).mean()  

      #run 1 iter of EM
      gmm_new = GMM_EM_iteration(X, gmm_old, psiEig=psiEig)

      #compute new log likelihood
      GMM_ll_new = logpdf_GMM(X, gmm_new).mean()

      #for sure GMM_ll_new >= GMM_ll_old
      #stop if GMM_ll_new - GMM_ll_old < threshold_stop
      if GMM_ll_old > GMM_ll_new:
         print("Warning: mean GMM log likelihood decreased. This is unexpected.")
         print(f"GMM_ll_old (mean): {GMM_ll_old}, GMM_ll_new (mean): {GMM_ll_new}")
         
      if GMM_ll_new - GMM_ll_old < threshold_stop:
         break

      num_iters += 1
      if num_iters >= max_iter:
         print(f"Reached maximum number of iterations: {max_iter}. Stopping EM.")
         break

      #update old GMM params
      gmm_old = gmm_new.copy()


   return gmm_new, GMM_ll_new  #return the last GMM parameters after EM iterations, plus the final log likelihood


def LBG(gmmToSplit, alpha):
    """
    Apply Linde-Buzo-Gray (LBG) algorithm to split a GMM component into two.
    This atomically splits a 1-Component GMM (i.e. a single Gaussian) into two components (i.e. two Gaussians).
    Parameters
    - gmmToSplit: a GMM with a single component, i.e. a tuple of (weight, mean, covariance).
                 weight: scalar
                 mean: vector of size (D,)
                 covariance: matrix of size (D, D)
    - alpha: scalar, the factor by which to scale the displacement vector. It dictates how far the two new components will be displaced from the original component.
    """

    #cov matrices are left like this
    weights_new = gmmToSplit[0] / 2  #split the weight in half

    #to separate the two components, we displace the mean of the original component along the direction of maximum variance
    #so in practice, we compute two means, which are the top of the Gaussian bell, and decide how far apart they have to be from each other
    #alpha is a scaling factor that determines how far apart the two new components will be from the original component
    #compute displacement vector by taking the leading eigevctor of the covariance matrix
    #in practce, we are displacing the 2 components along the direction of maximum variance
    #the step we use is the square root of the leading eigenvalue, which is the standard deviation along the leading eigenvector
    U, s, Vh = np.linalg.svd(gmmToSplit[2])  #SVD to get the leading eigenvector
    displacement_vector = vcol(U[:, 0] * np.sqrt(s[0]) * alpha)  #leading eigenvector scaled by the square root of the leading eigenvalue and alpha

    return [
        (weights_new, gmmToSplit[1] - displacement_vector, gmmToSplit[2]),  #first component
        (weights_new, gmmToSplit[1] + displacement_vector, gmmToSplit[2])   #second component
    ]  #return the two new components as a list of tuples

def train_GMM_EM_LBG(X, targetNumberComponents, threshold_stop=1e-6, alpha=0.1, psiEig=None):
    """
    Train a GMM using Expectation-Maximization algorithm, with a target number of components using the Linde-Buzo-Gray (LBG) algorithm.
    Parameters
    - X: matrix of size (D, N) where D is the number of features and N is the number of data points.
    - targetNumberComponents: target number of Gaussian components to reach in the GMM. It has to be either 1 or an even number, grater or equal than 2
    - threshold_stop: threshold for stopping the EM algorithm. If the change in log likelihood is less than this value, stop.
    - alpha: scalar, the factor by which to scale the displacement vector in the LBG algorithm.
            It dictates how far the two new components will be displaced from the original component, along the direction of maximum variance.
    - psiEig: optional, lowere bound > 0, to conntrain the eigenvalues of the covariance matrices of the GMM components to be > psiEig. This way, we won'threshold_stop
    have unboujnded GMM objective function for >=2 components (the GMM objective optimization in fact is an ill-posed problem, so we need to constrain the covariance matrices).
    Returns
    - gmm: list of gaussian components. Each one is a tuple of (weight, mean, covariance).
          weight: scalar
          mean: vector of size (D,)
          covariance: matrix of size (D, D)
    """
    #check if targetNumberComponents is either 1 or even, and greater than or equal to 2
    if targetNumberComponents != 1 and (targetNumberComponents < 2 or targetNumberComponents % 2 != 0):
        raise ValueError("targetNumberComponents must be either 1 or an even number greater than or equal to 2.")
    
    currentNumberComponents = 1
    GMM_ll_new = 0 #final log likelihood to be returned

    #start with a 1-component GMM, i.e. the initial GMM is a single Gaussian
    #its params, following a ML approach, are the empirical mean and empirical covariance matrix 
    mu, c = compute_mu_C(X)

    #if psiEig is provided, we have to contrain te eigvalues ALSO OF THE FIRST, STARTING COMPONENT
    if psiEig is not None:
        c = smooth_Covariance_Matrix(c, psiEig)  #contrain the covariance matrix eigenvalues to be > psiEig

    gmm_old = [(1.0, mu, c)]  #initial GMM with a single component, weight = 1.0, mean = mu, covariance = c

    while currentNumberComponents < targetNumberComponents:

        #phase 1: split components, go from G to 2G components
        gmm_new = []
        for k in range(currentNumberComponents):
            _2GComponents = LBG(gmm_old[k], alpha)  #split each component into two with LBG
            gmm_new.append(_2GComponents[0])  
            gmm_new.append(_2GComponents[1])
            currentNumberComponents += 1 #from 1 we split into 2, so we increase the number of components by 1 every time we split a component

        

        #phase2: use the 2G components as initial GMM for EM and let it converge
        gmm_old, GMM_ll_new = train_GMM_EM(X, gmm_new, threshold_stop=threshold_stop, psiEig=psiEig)


    return gmm_old, GMM_ll_new  #return the final GMM parameters after EM iterations, plus the final log likelihood

    
