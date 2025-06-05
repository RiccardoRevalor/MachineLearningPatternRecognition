import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

irisPath = "iris.csv"


def loadIrisDataset(path):
    '''
    Params:
    - path: filename of the dataset to load

    Returned Values:
    - D: Matrix of dimensions (num_features, num_samples) of the dataset
    - L: Array of the labels for each one of the dataset samples
    '''
    rawData = np.genfromtxt(path, delimiter=',', dtype = "str")
    
    #extract the 4 features and insert them into D
    D = rawData[:, 0:4] #D shape: (150, 4)
    D = np.array(D, np.float32).T  #Dont't do reshape! Because I want onr row= 150 samples of the same feature! .reshape((numFeatures, numSamples))

    #L_string = rawData[:, 4].reshape((numSamples, 1))  
    classLabels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    
    L = np.array([classLabels[name] for name in rawData[:, 4]], dtype=int) #.reshape((numSamples, 1)) #I want a column vector of shape (150, 1)
    

    return D, L



#Compute Sw, Sb

def computeSwSb(D, L):
    '''
    Params:
    - D: Dataset features matrix, not ceCntered
    - L: Labels of the samples

    Returned Values:
    - Sw: Within-class scatter matrix
    - Sb: Between-class scatter matrix
    '''

    #find the unique labels for each class
    uniqueLabels = np.unique(L)

    #nc in the formula is computed as the number of samples of class c
    #separate data into classes
    DC = [D[:, L == label] for label in uniqueLabels]  #DC[0] -> samples of class 0, DC[1] -> samples of class 1 etc...

    #compute nc for each class
    #each element in DC has a shape which is (4, DC_i.shape[1]) (assuming samples are not equally distributed among all the classes which is true in 99% of cases...)
    #So for nc I just have to take DC_i.shape[1] for each i in DC
    nc = [DC_i.shape[1] for DC_i in DC]

    #Compute the mean as done before with PCA
    mu = D.mean(axis=1)
    mu = mu.reshape((mu.shape[0], 1))

    #Now compute the mean for each class
    muC = [DC[label].mean(axis=1) for label, labelName in enumerate(uniqueLabels)]
    muC = [mc.reshape((mc.shape[0], 1)) for mc in muC]

    Sb = 0  #between matrix initialization
    Sw = 0  #within  matrix initialization

    #iterate over all the classes to execute the summations to calculate the 2 matrices
    for label, labelName in enumerate(uniqueLabels):

        #1) FOR SB:
        #add up to the Sb (between) matrix
        diff = muC[label] - mu
        Sb += nc[label] * (diff @ (diff.T)) #nc * ((muC - mu) * (muC - mu).T)


        #2) FOR SW
        #add up to the Sw (within) matrix
        #for diff1 subtract the the class mean from the samples of each class, i.e center center the samples for each class 
        diff1 = DC[label] - muC[label]  #x_{c, i} - muC done by rows

        #SHORTCUT: compute the Sw matrix as a weighted sum of the covariance matrices of each class
        #so for each class:
        #Compute the Covariance Matrix C using DC = D - mu
        C_i = (diff1 @ diff1.T) / float(diff1.shape[1])  #Covariance matrix for class i

        #weighted sum of all the C_i
        Sw += nc[label] * C_i

    
    #at the end of the summations, just multiply by 1/N (N is the number of samples)

    Sb = Sb / D.shape[1]
    Sw = Sw / D.shape[1]

    #return both matrices
    return Sb, Sw
        





# FIRST METHOD 
def calculateLDA(D, L, m):
    '''
    Params:
    - D: Dataset features matrix, not ceCntered
    - L: Labels of the samples
    - m: number of discriminant directions of the target subplane

    Returned Values:
    - W: LDA matrix containing the most discriminant directions, which are the m leading Eigenvectors found solving the generalized Eigenvalues problem for matrices Sb, Sw
    '''

    #1) first, compute S between, S within matrices
    Sb, Sw = computeSwSb(D, L)

    #2) then, solve the generalized Eigenvalues problem
    #matrix a is Sb -> I wanna find the eigvalues of (Sw)^{-1} * Sb 
    #matrix b is Sw which is positive definite

    #sigma -> w ndarray of eigenvalues of inverse of matrix b * matrix a
    #U -> v ndarray of corresponding eigenvectors of inverse of matrix b * matrix a -> they are sorted from lowest to highest as in the numpy counterpart!
    sigma, U = sc.linalg.eigh(a=Sb, b=Sw)

    #I don't mind about sigma since I want the LDA directions which are the eigenvectors of U
    #Take the leading m eigenvectors of U to create matrix W of LDA main directions (first I have to sort U from highest to lowest so the other way around):
    W = U[:, ::-1][:, 0:m]

    return W


# SECOND METHOD
def calculateLDAbyJointDiagonalization(D, L, m):
    '''
    Params:
    - D: Dataset features matrix, not ceCntered
    - L: Labels of the samples
    - m: number of discriminant directions of the target subplane

    Returned Values:
    - W: LDA matrix containing the most discriminant directions, calculated solving teh generalized Eigenvalue problem by joint diagonalization of Sw and Sb
    '''

    #0) first, compute S between, S within matrices as before
    Sb, Sw = computeSwSb(D, L)

    #Joint diagonaization:

    #Step 1: Whiten Sw
    #Compute SVD on matrix Sw to find U and sigma (Vh is useless)
    U, sigma, Vh = np.linalg.svd(Sw)

    #U = matrix of singular vectors of Sw
    #sigma = matrix of singular values of Sw

    #Compute P1
    #sigma needs to be to the power of -1/2 first, then use dot products: 
    #a. compute U @ sigma^{1/2}
    #b. compute (U @ sigma^{1/2}) @ U.T
    P1 = U @ np.diag(1.0/(sigma**0.5)) @ U.T

    #Compute Sbt = Sbt = P1 @ Sb @P1 -> Sbt is the transformed between class covariance
    Sbt = P1 @ Sb @ P1.T

    #Compute P2, which is the matrixof eigenvectors of Sbt corresponding to its m highest eigenvalues -> so use SVD again
    U2, sigma2, Vh2 = np.linalg.svd(Sbt) #with SVD the eigvectors are already ordered from the highest to the lowest

    #just take the leading m eigvectors
    P2 = U2[:, 0:m]



    #Step 2: Find matrix W as W = P1.T @ P2
    return P1.T @ P2


#THIS (THE FINAL PROJECTION STEP) IS EXACTLY THE SAME FOR BOTH METHODS
def applyLDA(D, W):
    '''
    Params:
    - D: Dataset features matrix, not ceCntered
    - W: LDA matrix containing the most discriminant directions, which are the m leading Eigenvectors found solving the generalized Eigenvalues problem for matrices Sb, Sw

    Returned Values:
    - W: The projection of the dataset onto the subplane having the m most discriminant directions
    '''

    return W.T @ D


def scatterPlots2(x, y, numFeatures):
    #I select feature i and feature j, with i different from j and plot them on the 2 axis of every scatter plot chart

    #rows, cols for making the subplot
    cols = 2
    rows = 2

    labelColors = {0: "red", 1: "green", 2: "blue"}
    classLabels = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    featuresNames = {0: 'Sepal length', 1: 'Sepal width', 2: 'Petal length', 3: 'Petal width'}

    #subplot creation
    fig, plots = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*6, rows*4))
    plots = plots.flatten()   #the 2D axes array becomes a 1D array in order to access each ax in a more straighforward way during the loop

    subplotCounter = 0
    for i in range(numFeatures):
        for j in range(numFeatures):
            if j == i: continue

            #choose the subplot
            subplot = plots[subplotCounter]
            subplotCounter+=1

            #plt.scatter accepts x and y vectors
            for label in labelColors:
                xFeature = x[i][ y == label]
                yFeature = x[j][ y == label]
                subplot.scatter(xFeature, yFeature, color= labelColors[label], alpha=0.7,  label=f"{classLabels[label]}", edgecolor="black")

            subplot.set_xlabel(featuresNames[i])
            subplot.set_ylabel(featuresNames[j])
            subplot.legend()
            subplot.set_title(f"Scatter plot: {featuresNames[i]} vs {featuresNames[j]}")

        subplot = plots[subplotCounter]
        subplotCounter+=1
        for label in labelColors:
            sample_with_that_class = x[i][ y == label]
            subplot.hist(x=sample_with_that_class, color=labelColors[label], alpha= 0.7, density=True, label=f"{classLabels[label]}", edgecolor="black")
            subplot.legend()
            subplot.set_title(f"Feature {featuresNames[i]} Distribution")
            subplot.set_xlabel(featuresNames[i])
            subplot.set_ylabel("Density")

    plt.tight_layout(pad=3) #add padding between subplots to distance between eachother
    plt.show()


def densityPlots(x, y, numFeatures, title=None):
    #I select feature i and feature j, with i different from j and plot them on the 2 axis of every scatter plot chart

    #rows, cols for making the subplot
    cols = 2
    rows = 1

    labelColors = {0: "red", 1: "green", 2: "blue"}
    classLabels = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    featuresNames = {0: 'Sepal length', 1: 'Sepal width', 2: 'Petal length', 3: 'Petal width'}

    #subplot creation
    fig, plots = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*6, rows*4))
    plots = plots.flatten()   #the 2D axes array becomes a 1D array in order to access each ax in a more straighforward way during the loop

    subplotCounter = 0
    for i in range(numFeatures):

        subplot = plots[subplotCounter]
        subplotCounter+=1
        for label in labelColors:
            sample_with_that_class = x[i][ y == label]
            subplot.hist(x=sample_with_that_class, color=labelColors[label], alpha= 0.7, density=True, label=f"{classLabels[label]}", edgecolor="black")
            subplot.legend()
            subplot.set_title(f"Feature {featuresNames[i]} Distribution")
            subplot.set_xlabel(featuresNames[i])
            subplot.set_ylabel("Density")

    plt.tight_layout(pad=3) #add padding between subplots to distance between eachother
    if title:
        fig.suptitle(title, fontsize=16)  # General title
    plt.show()


if __name__ == "__main__":
    D, L = loadIrisDataset(irisPath)

    m = 2   #number of most discriminant directions (at max equal to C-1)

    #Method 1, Standard solution to the generalized Eigvalue problem
    W_2_Method1 = calculateLDA(D, L, m)

    #Apply Method 1
    DP_2_Method1 = applyLDA(D, W_2_Method1)

    
    


    #Method 2 - Joint Diagonalization of Sb, Sw
    W_2_JointDiag = calculateLDAbyJointDiagonalization(D, L, m)

    #Apply Method 2
    DP_2_JointDiag = applyLDA(D, W_2_JointDiag)

    
    
    
    
    #Plot both results

    scatterPlots2(DP_2_Method1, L, DP_2_Method1.shape[0])   #Method 1
    
    scatterPlots2(DP_2_JointDiag, L, DP_2_JointDiag.shape[0])   #Joint Diag
    

 





