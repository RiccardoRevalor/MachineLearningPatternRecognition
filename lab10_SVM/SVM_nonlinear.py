import numpy as np
from mean_covariance import vcol, vrow
import scipy.optimize as opt
from decisionModelsEvaluation import computeEmpiricalBayesRisk_Normalized, computeMinEmpiricalBayesRisk_Normalized

#wrappers to compute kernel functions

def get_poly_kernel_function(d, c):
    """
    :param d: degree of the polynomial
    :param c: constant
    """
    def poly_kernel(xi, xj):
        #compute the polynomial kernel
        #DTR = (F, N) -> xj
        #DTR.T = (N, F) -> xi^T
        #<xi, xj> = xi^T xj = DTR.T @ DTR = (N, F) * (F, N) = (N, N)
        k = (xi.T @ xj + c) ** d
        return k
    

    #return the FUNCTION of the poly kernel
    #so, function of xi and xj
    return poly_kernel


def get_rbf_kernel_function(sigma):
    """
    :param sigma: standard deviation of the Gaussian kernel.
    """

    def rbf_kernel(xi, xj):
        #the issue is that we have to compute pairwise distances between xi and xj
        #method 1: for loops
        """
        k_func = np.zeros((xi.shape[1], xj.shape[1])) #(N, N)
        for i in range(xi.shape[1]):
            for j in range(xj.shape[1]):
                x1 = xi[:, i:i+1] #here it's not important if xi is (F, N) or (N, F), since the norm is the same, it just has to have the same shape of xj
                x2 = xj[:, j:j+1]
                squaredNormDistance = np.linalg.norm(x1 - x2)**2
                k_func[i, j] = np.exp(-1 * sigma * squaredNormDistance)

        return k_func
        """
        #method 2: explot squared norm properties -> WAY MORE EFFICIENT
        #1. norm(xi - xj)**2 = <xi - xj, xi - xj> 
        #2. <xi - xj, xi - xj> = (x- xj)^T * (xi - xj)
        #3. (xi - xj)^T * (xi - xj) = xi^T * xi - xi^T * xj - xj^T * xi + xj^T * xj
        #4. xi^T * xi = ||xi||^2
        #5. xj^T * xj = ||xj||^2
        #6. for real vectors, the dot product is symmetric, so xi^T * xj = xj^T * xi
        #7. so, we can compute the squared norm distance as:
        #xi^T * xi + xj^T * xj - 2 * xi^T * xj = ||xi||^2 + ||xj||^2 - 2 * xi^T * xj

        
        #use axis=0 to compute the norm along the columns i.e. for each sample we compute the norm
        #xi = (F, N) 
        #xj = (F, N)
        #for each of the N columns, it will compute the norm of that column vector (which has F elements):
        #np.linalg.norm(xi, axis=0)**2 = (N, )
        #np.linalg.norm(xj, axis=0)**2 = (N, )
        #2 * (xi.T @ xj) = (N, F) * (F, N) = (N, N)
        #BEWARE: (N, ) - (N, N) is NOT COMPATIBLE, so we need to use broadcasting
        #I can do: vcol(np.linalg.norm(xi, axis=0)**2) + vrow(np.linalg.norm(xj, axis=0)**2) - 2 * (xi.T @ xj)
        #because: vcol(np.linalg.norm(xi, axis=0)**2) = (N, ) -> (N, 1)
        #vrow(np.linalg.norm(xj, axis=0)**2) - 2 * (xi.T @ xj) = (N, ) -> (1, N)
        #so their sum, with broadcasting, will be (N, N)
        #(N, 1) + (1, N) = (N, N)
        #(N, N) - (N, N) = (N, N) OK!
        squaredNormDistance = vcol(np.linalg.norm(xi, axis=0)**2) + vrow(np.linalg.norm(xj, axis=0)**2) - 2 * (xi.T @ xj)
        k = np.exp(-1 * sigma * squaredNormDistance)
        return k
    
    
    #return the FUNCTION of the rbf kernel
    #so, function of xi and xj
    return rbf_kernel



def train_SVM_Kernel_SoftMargin_Dual(DTR, LTR, C, kernel, K = 1):
    #compute ZTR -> z_i = 2* c_i - 1
    ZTR = 2 * LTR - 1
    #compute n-dimensional vector of ones
    Ones = np.ones((DTR.shape[1], 1))
    #compute xi
    xi = K**2

    #since H is shared by both primal and dual, we compute it outside
    #compute H = z_i * z_j * kernal_hat(xi, xj)
    #kernal_hat(xi, xj) = kernel(xi, xj) + xi
    kernel_hat = kernel(DTR, DTR) + xi
    #use broadcasting to compute z_i * z_j -> vcol(z_i) * vrow(z_j)
    #
    #(N, N) * [(1, N) * (N, 1)] = (N, N) * (N, N) = (N, N)
    #H = kernel_hat * (ZTR.reshape((1, ZTR.size)) * ZTR.reshape((ZTR.size, 1)))
    H = kernel_hat * (vrow(ZTR) * vcol(ZTR)) #use * and not @ because it's an ELEMENT WISE product -> H_i,j = z_i * z_j * x_i^T * x_j -> ELEMENT WISE!!!


    def SVM_dualObj(alpha):
        """
        Dual Objective function for SVM
        :param alpha: Lagrange multipliers vector -> (N, ) = (1, N)
        :return: objective function value
        """


        #1) H is computed boutsise since it's shared both by dual obj and primal obj!

        #2) compute the objective function value
        #vrow(alpha) * H * vcol(alpha)
        #(1, N) * (N, N) * (N, 1) = (1, 1) -> SCALAR
        #(1, N) * (N, N) = (1, N)
        #(1, N) * (N, 1) = (1, 1)
        #first_term = 0.5 * (alpha.reshape((1, alpha.size)) @ H @ alpha.reshape((alpha.size, 1)))
        first_term = 0.5 * (vrow(alpha) @ H @ vcol(alpha)).ravel() #ravel() to convert to 1D array
        #since I've used the dot product, numpy acually returns a (1, 1) array, so I need to convert it to a scalar
        first_term = first_term.item()

        #(1, N) * (N, 1) = (1, 1)
        #second_term = - alpha.reshape((1, alpha.size)) @ Ones #this is the dot product between alpha and a vector of all ones -> this is equal to np.sum(alpha)
        second_term = - (vrow(alpha) @ Ones)
        #since I've used the dot product, numpy acually returns a (1, 1) array, so I need to convert it to a scalar
        second_term = second_term.item()

        #3) manually compute the gradient
        #(N, N) * (N, 1) = (N, 1)
        #Ones is (N, 1)
        #gradient = H @ alpha.reshape((alpha.size, 1)) - Ones
        gradient = (H @ vcol(alpha) - Ones).ravel()

        return first_term + second_term, gradient
    
    #box contraints
    box_contraints = (0, C)
    bounds = [box_contraints] * DTR.shape[1]

    #find alpha which maximizes the dual objective function 
    #since we are using L-BFGS-B, we need to minimize the negative of the objective function
    #factr and pgtol are used to control the convergence of the algorithm
    bestAlpha_hat, _, info = opt.fmin_l_bfgs_b(func= SVM_dualObj, x0 = np.zeros(DTR.shape[1]), approx_grad=False, bounds=bounds, factr=np.nan, pgtol=1e-5, maxfun=20000)

    times_objCalled = info['funcalls']
    iters = info['nit']
    print(f"\nDual objective function called {times_objCalled} times, number of iterations: {iters}")
    


    #now check the duality gap
    def SVM_primalObj(alpha):
        #the first term is 0.5 * alpha^T * H * alpha
        #alpha = (N, )
        #H = (N, N)
        first_term = 0.5 * (vrow(alpha) @ H @ vcol(alpha)).ravel() #ravel() to convert to 1D array
        #since I've used the dot product, numpy acually returns a (1, 1) array, so I need to convert it to a scalar
        first_term = first_term.item()

        #second term

        #np.maximum(array1, array2, ...) is faster than np.amax([array1, array2, ..], axis=0), so use np.maximum
        #H = (N, N)
        #alpha = (N, )
        #H @ vcol(alpha) = (N, N) * (N, 1) = (N, 1)
        #0 and 1 are not scalars, they are broadcasted
        #H_{i, *} represents the i-th row of the matrix H. This is a row vector, say of shape (1, N)
        #When you compute the matrix-vector product Hα, you get a column vector of shape (N, 1)
        #this contains all the H_{i, *} as its rows
        #so at the end we have a column vector of shape (N, 1)
        #we take the max
        #then we need to sum over all samples
        second_term = np.maximum(0, 1 - H @ vcol(alpha))
        #sum over all samples and multiply by constant C
        second_term = C * np.sum(second_term)

        return first_term + second_term
    

    #compute duality gap
    primal_optimum = SVM_primalObj(bestAlpha_hat)
    dual_optimum = SVM_dualObj(bestAlpha_hat)
    #in this case, since with the L-BFGS-B we are minimizing the negative of the dual objective function
    #dua_optimum will be negative, so do: dualityGap = primal_optimum + dual_optimum[0]
    dualityGap = primal_optimum +  dual_optimum[0] #do not take the gradient, just the function of the dual
    
    print(f"SVM_Linear_SoftMargin, hyperparams C = {C}, K = {K}, Primal(bestAlpha_hat) = {primal_optimum}, Dual(bestAlpha_hat) = {-dual_optimum[0]}, computed duality gap: {dualityGap}")

    #return also bestAlha_hat, it will be needed for scores computation
    return bestAlpha_hat, primal_optimum, dual_optimum[0], dualityGap
    
        


def fit_SVM_Kernel_SoftMargin(DTR, LTR, DVAL, LVAL, C, kernel, K = 1, appPriorTrue=0.5):
    """
    Train and fit the SVM Linear Soft Margin classifier
    """
    ZTR = 2 * LTR - 1
    #compute kernel function using both DTR and DVAL
    #so, for each DVAL sample, we compute the kernel function with the DTR support vectors (for which alpha is > 0)
    kernel_hat = kernel(DTR, DVAL) + K**2 #add xi to the kernel matrix
    bestAlpha_hat, primal_loss, optimal_loss, dualitygap = train_SVM_Kernel_SoftMargin_Dual(DTR, LTR, C, kernel, K)

    #compute the scores
    #scores = sum_i (alpha_i * z_i * kernel_hat(xi, xj))
    #alpha_i = (N, )
    #ZTR = (N, )
    #kernel_hat = (N, N)
    scores = (vrow(bestAlpha_hat * ZTR) @ kernel_hat).ravel() #ravel() to convert to 1D array

    #decision rule: assign class H_T if score > 0, otherwise assign class H_F
    PVAL = (scores > 0) * 1

    #calculate error rate
    errorRate = np.mean(LVAL != PVAL)

    #calculate DCF, min DCF using appPriorTrue
    minDCF = computeMinEmpiricalBayesRisk_Normalized(scores, LVAL, appPriorTrue, 1.0, 1.0)
    DCF = computeEmpiricalBayesRisk_Normalized(scores, LVAL, appPriorTrue, 1.0, 1.0)


    return bestAlpha_hat, minDCF, DCF, errorRate, primal_loss, optimal_loss, dualitygap


    
