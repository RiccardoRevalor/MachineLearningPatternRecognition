import numpy as np
import matplotlib.pyplot as plt

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



def computeC(D):
    '''
    Params:
    - D: Dataset features matrix, not centered

    Returned Values:
    - C: Covariance matrix of centered D
    '''
    
    mu = D.mean(axis = 1)                       #compute mu as a row array
    mu = mu.reshape((mu.shape[0], 1))           #reshape mu as a column array
    DC = D - mu                                 #center D
    C = (DC @ DC.T) / float(DC.shape[1])        #compute C, the Covariance Matrix, from DC

    return C


def calculatePCA(D, m):
    '''
    Params:
    - m: dimensionality of target subspace (m has to be <= dimensionality of original space of D)
    - D: Dataset features matrix, not centered

    Returned Values:
    - P: the subspace whose dimensionality is m
    '''

    #1. Compute C, the Covariance Matrix
    C = computeC(D)

    #2. Use SVD since it's usually more efficient than Eigen Decomposition and cover a broaders range of cases
    #   U: left eigen vectors of A.T*A, so left singular vectors of A
    #   s = sigma -> singular values of A
    #   Vh = transpose of V -> right eigen vectors of A.T*A, so right singular vectors of A
    U, s, Vh = np.linalg.svd(C)

    #3. The Eigen Values are already sorted from the largest to the smallest
    #   Take just the leading m Eigen Vectors describing the subspace P
    P = U[:, 0:m]

    #4. Return the subspace P
    return P


def applyPCA(D, P):
    '''
    Params:
    - D: Dataset features matrix, not centered
    - P: Subspace onto which you want to project the samples D
    '''

    #using np.dot or @ is the same
    return P.T @ D  #return np.dot(P.T, D)


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


if __name__ == "__main__":
    D, L = loadIrisDataset(irisPath)

    m = 2   #target number of imensionalities of the subspace P

    P = calculatePCA(D, m)  #target subspace 

    # Apply PCA and return the samples projected onto P
    DP = applyPCA(D, P)

    # Scatter Plot of the 2 components
    scatterPlots2(DP, L , DP.shape[0])

