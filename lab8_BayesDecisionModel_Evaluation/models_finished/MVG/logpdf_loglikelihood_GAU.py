import numpy as np
import matplotlib.pyplot as plt


#Functions to compute Gaussian log probability density function

def logpdf_GAU_ND_singleSample(x, mu, C):
    #Compute the log of the probability density function of a multivariate Gaussian distribution
    """
    Parameters
    - x: numpy array of shape (M, 1) -> it's a single sample having M features.
         x is a column vector!
    - mu: numpy array of shape (M,) -> mean of the distribution
    - C: numpy array of shape (M,M) -> covariance matrix of the distribution

    Returned values:
    - logpdf: float -> the log of the probability density function of the multivariate Gaussian distribution
    """

    #Since x is a columns vector of shape (M, 1):
    M = x.shape[0]  #M is the number of features

    #Compute te inverse of the covariance matrix
    C_inv = np.linalg.inv(C)

    #Compute the log of the determinant of the covariance matrix
    #the function np.linalg.slogdet returns a tuple (sign, logdet)
    #the first element of the tuple is the sign of the determinant, which is positive for covariance matrices, and which is not needed here
    #the second element of the tupe is the log determinant of C
    C_logDet = np.linalg.slogdet(C)[1]

    #return -0.5 * M * np.log(2*np.pi) - 0.5 *C_logDet - 0.5 * (x-mu).T @ C_inv @ (x-mu)
    return -0.5 * (M * np.log(2*np.pi) + C_logDet + (x-mu).T @ C_inv @ (x-mu))


def logpdf_GAU_ND_Loop(X, mu, C):
    #Compute the log of the probability density function of a multivariate Gaussian distribution
    """
    Parameters
    - X: numpy array of shape (M, N) -> it contains multiple samples [x1, x2, ..., xN] having M features.
         X is a matrix of shape (M, N)!
    - mu: numpy array of shape (M,) -> mean of the distribution
    - C: numpy array of shape (M,M) -> covariance matrix of the distribution

    Returned values:
    - logpdf: float -> the log of the probability density function of the multivariate Gaussian distribution
    """

    #Attempt 1: Using a loop
    #Loop over all teh samples and for each sample call the function logpdf_GAU_ND_singleSample
    N = X.shape[1]
    result = []

    for i in range(N):
        #print("Shape of X[:, i]:", X[:, i].shape) #-> it must always be (M,) 
        result.append(logpdf_GAU_ND_singleSample(X[:, i], mu, C))

    #return a numpy array and not a list
    """
    The list is structured as follows, for ex:
    [[[-21.51551212]],

     [[-21.42552223]],

     ....]

     So I have to flatten it to get a 1D array. Do I use flatten() or ravel()?
    - flatten is a method of an ndarray object and hence can only be called for true numpy arrays.
    - ravel is a library-level function and hence can be called on any object that can successfully be parsed.

     For example ravel will work on a list of ndarrays, while flatten is not available for that type of object.
     In this case, both will work, but ravel is more general and hence recommended.
    """
    
    #At the end, I return a np array of shape (N, 1) because I iterated N times (because I got N samples) and for each iteration I got a scalar value
    
    return np.array(result).ravel()


def logpdf_GAU_ND(X, mu, C):
    #Compute the log of the probability density function of a multivariate Gaussian distribution
    """
    Parameters
    - X: numpy array of shape (M, N) -> it contains multiple samples [x1, x2, ..., xN] having M features.
         X is a matrix of shape (M, N)!
    - mu: numpy array of shape (M,) -> mean of the distribution
    - C: numpy array of shape (M,M) -> covariance matrix of the distribution

    Returned values:
    - logpdf: float -> the log of the probability density function of the multivariate Gaussian distribution
    """

    #Attempt 2: Using vectorized operations, broadcasting, without loops
    #Let me think: matrix X has a shape of (M, N)...I need to compute the logpdf for each sample, so I need to compute the logpdf for each column of X
    #What can I do for achieving this without using a loop to scan all the columns of X?
    
  
    M = X.shape[0]  

    C_inv = np.linalg.inv(C)

    C_logDet = np.linalg.slogdet(C)[1]

    #The problems is related to the term; (X-mu).T @ C_inv @ (X-mu)
    #I need to compute this term for each column of X
    #If i Just write it like this, this operation will be wrong because will result in an overall result of shape (M, M) since the operation will be done element-wise 
    #But my goal is to do it column wise!
    #with (X-mu).T @ C_inv @ (X-mu) = (X_centered).T @ C_inv @ X_centered
    #I have: (M,N)⋅(M,M)⋅(M,N)⇒(N,M)⋅(M,N)=(N,N)
    #C_inv @ X_centered has a shape of (M, N)


    #Compute X centered using broadcasting 
    X_centered = X - mu #Shape (M, N)

    #print(f"Shape of X_centered: {X_centered.shape}")
  
    
    #I can obtain the same final shape of (N, N) by doing like this:
    #1. C_inv @ X_centered has a shape of (M, N)
    #2. X_centered.T has a shape of (N, M)
    #3. X_centered * (C_inv @ X_centered) is an element wise multiplication of two matrices of shape (M, N) and (N, M) and the result is a matrix of shape (N, M)
    #4. Then I can just sum over the columns (= I sum all the rows) of the resulting matrix to get a vector of shape (N,)
    
    #quadratic_terms = np.sum(X_centered * (C_inv @ X_centered), axis=0)
    quadratic_terms = np.einsum('ij,ji->i', X_centered.T, C_inv @ X_centered)
    #print(f"Shape of quadratic_terms: {quadratic_terms.shape}")

    return (-0.5 * (M * np.log(2*np.pi) + C_logDet + quadratic_terms))



def plotPdf_compute(XPlot, m, C, plot_hist_Xplot=False, plot_hist_Xplot_bins=50):
    """
    Compute the log of the probability density function of a multivariate Gaussian distribution and plot it
    Parameters:
    - XPlot: numpy array of shape (1, N) -> it contains the values of the x-axis where the pdf will be plotted
    - m: numpy array of shape (M,) -> mean of the distribution
    - C: numpy array of shape (M,M) -> covariance matrix of the distribution
    - plot_hist_Xplot: boolean -> if True, the histogram of XPlot will be plotted, in order to see how well the pdf fits the data
    - plot_hist_Xplot_bins: int -> number of bins for the histogram of XPlot
    """
    plt.figure()
    if plot_hist_Xplot:
        plt.hist(XPlot.ravel(), bins=plot_hist_Xplot_bins, density=True)

    #I use the exponential scale to plot the logpdf, so I end up plotting the exp(log(pdf)) = pdf
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot, m, C)) )
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.title('pdf of a Gaussian distribution over sample data')
    plt.grid()
    plt.show()

def plotPdf(Xplot, logpdf, plot_hist_Xplot=False, plot_hist_Xplot_bins=50):
    """
    Plot the pdf of a Gaussian distribution
    Parameters:
    - Xplot: numpy array of shape (1, N) -> it contains the values of the x-axis where the pdf will be plotted
    - logpdf: numpy array of shape (1, N) -> it contains the log of the probability density function of the multivariate Gaussian -> ALREADY COMPUTED
    """
    plt.figure()

    if plot_hist_Xplot:
        plt.hist(Xplot.ravel(), bins=plot_hist_Xplot_bins, density=True)

        
    #I use the exponential scale to plot the logpdf, so I end up plotting the exp(log(pdf)) = pdf
    plt.plot(Xplot.ravel(), np.exp(logpdf) )
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.title('pdf of a Gaussian distribution over sample data')
    plt.grid()
    plt.show()


#from mpl_toolkits.mplot3d import Axes3D

def plotPdf3D_compute(XPlot, m, C):
    """
    Variant of the function plotPdf_compute that plots the pdf in 3D
    To be used when the dataset is multidimensional and so we compute the Multivariate Gaussian distribution
    Compute the log of the probability density function of a multivariate Gaussian distribution and plot it
    Parameters:
    - XPlot: numpy array of shape (1, N) -> it contains the values of the x-axis where the pdf will be plotted
    - m: numpy array of shape (M,) -> mean of the distribution
    - C: numpy array of shape (M,M) -> covariance matrix of the distribution
    """
    logpdf = logpdf_GAU_ND(XPlot, m, C)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XPlot[0, :], XPlot[1, :], np.exp(logpdf), c=np.exp(logpdf), cmap='viridis')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('PDF')
    ax.set_title('3D PDF of a Gaussian distribution')
    plt.show()


def plotPdf3D(XPlot, logpdf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XPlot[0, :], XPlot[1, :], np.exp(logpdf), c=np.exp(logpdf), cmap='viridis')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('PDF')
    ax.set_title('3D PDF of a Multivariate Gaussian Distribution (MVG)')
    plt.show()

#Functions to compute the log likelihood of a multivariate Gaussian distribution
#The log likelihood is the sum of the log pdf of each sample, so it's a scalar value

def loglikelihood(X, mu, C):
    #Compute the log LIKEHOOD of the probability density function of a multivariate Gaussian distribution
    """
    Parameters
    - X: numpy array of shape (M, N) -> it contains multiple samples [x1, x2, ..., xN] having M features.
         X is a matrix of shape (M, N)!
    - mu: numpy array of shape (M,) -> mean of the distribution
    - C: numpy array of shape (M,M) -> covariance matrix of the distribution

    Returned values:
    - loglikelihood: float -> the log likelihood of the multivariate Gaussian distribution -> the sum of the logpdf of each sample, so it's a scalar value
    """

    #Compute the logpdf of the multivariate Gaussian distribution as done before
    #Then it's a function of shape (N,) where N is the number of samples
    #You can treat it a a discrete probability distribution and sum all the values to get the loglikelihood
    #Meaning, for each sample you take the logpdf[i] which in mathematical terms is the logpdf of the i-th sample, and then you sum all the logpdfs to get the loglikelihood

    logpdf = logpdf_GAU_ND(X, mu, C)

    """
    This method is worse in terms of complexity because it uses a for loop
    logLikelihood_acc = 0

    for i in range(X.shape[1]):
        logLikelihood_acc += logpdf[i]


    return logLikelihood_acc
    """

    return np.sum(logpdf)

def loglikelihood2(X, logpdf):
    #Compute the log LIKEHOOD of the probability density function of a multivariate Gaussian distribution
    """
    Parameters
    - X: numpy array of shape (M, N) -> it contains multiple samples [x1, x2, ..., xN] having M features.
         X is a matrix of shape (M, N)!
    - logpdf: numpy array of shape (1, N) -> it contains the log of the probability density function of the multivariate Gaussian -> ALREADY COMPUTED

    Returned values:
    - loglikelihood: float -> the log likelihood of the multivariate Gaussian distribution -> the sum of the logpdf of each sample, so it's a scalar value
    """
    return np.sum(logpdf)


   
    