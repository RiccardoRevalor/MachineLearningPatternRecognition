import numpy as np
from mean_covariance import vcol, vrow
import scipy.optimize as opt
from decisionModelsEvaluation import computeEmpiricalBayesRisk_Normalized, computeMinEmpiricalBayesRisk_Normalized


def train_SVM_Linear_SoftMargin_Dual(DTR, LTR, C, K = 1):
    #map DTR to DTR_hat = [DTR, K_vector]
    #stack verticallu (row wise)
    #so, at the end of DTR add an additional row of all K
    DTR_hat = np.vstack((DTR, K * np.ones((1, DTR.shape[1]))))
    #compute ZTR -> z_i = 2* c_i - 1
    ZTR = 2 * LTR - 1
    #compute n-dimensional vector of ones
    Ones = np.ones((DTR.shape[1], 1))


    def SVM_dualObj(alpha):
        """
        Dual Objective function for SVM
        :param alpha: Lagrange multipliers vector -> (N, ) = (1, N)
        :return: objective function value
        """


        #1) compute H_hat = z_i * z_j * x_i^T * x_j
        #G = x_i^T * x_j = <x_i, x_j>
        G = DTR_hat.T @ DTR_hat
        #use broadcasting to compute z_i * z_j -> vcol(z_i) * vrow(z_j)
        #
        #(N, N) * [(1, N) * (N, 1)] = (N, N) * (N, N) = (N, N)
        #H_hat = G * (ZTR.reshape((1, ZTR.size)) * ZTR.reshape((ZTR.size, 1)))
        H_hat = G * (vrow(ZTR) * vcol(ZTR)) #use * and not @ because it's an ELEMENT WISE product -> H_i,j = z_i * z_j * x_i^T * x_j -> ELEMENT WISE!!!

        #2) compute the objective function value
        #vrow(alpha) * H_hat * vcol(alpha)
        #(1, N) * (N, N) * (N, 1) = (1, 1) -> SCALAR
        #(1, N) * (N, N) = (1, N)
        #(1, N) * (N, 1) = (1, 1)
        #first_term = 0.5 * (alpha.reshape((1, alpha.size)) @ H_hat @ alpha.reshape((alpha.size, 1)))
        first_term = 0.5 * (vrow(alpha) @ H_hat @ vcol(alpha)).ravel() #ravel() to convert to 1D array
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
        #gradient = H_hat @ alpha.reshape((alpha.size, 1)) - Ones
        gradient = (H_hat @ vcol(alpha) - Ones).ravel()

        return first_term + second_term, gradient
    
    #box contraints
    box_contraints = (0, C)
    bounds = [box_contraints] * DTR_hat.shape[1]

    #find alpha which maximizes the dual objective function 
    #since we are using L-BFGS-B, we need to minimize the negative of the objective function
    #factr and pgtol are used to control the convergence of the algorithm
    bestAlpha_hat, _, info = opt.fmin_l_bfgs_b(func= SVM_dualObj, x0 = np.zeros(DTR_hat.shape[1]), approx_grad=False, bounds=bounds, factr=np.nan, pgtol=1e-5, maxfun=20000)

    times_objCalled = info['funcalls']
    iters = info['nit']
    print(f"\nDual objective function called {times_objCalled} times, number of iterations: {iters}")
    #then we can recover best_w_hat using the primal contraint, so the I KKT:ù
    #w_hat = sum_i (bestAlpha * z_i * x_i)
    #alpha: (N, )
    #ZTR: (N, )
    #alpha * ZTR = (N, )
    #DTR_hat: (F + 1, N)
    #alpha times DTR -> matrix vector multiplication -> I can explicitly tell to treat alpha * ZTR as a column vector (but it should be automatic)
    #bestW_hat = (F + 1, N) * (N, 1) = (F + 1, 1)           (F = features, N = samples)
    bestW_hat = DTR_hat @ vcol(bestAlpha_hat * ZTR)
    #this is equivalent to: (DTR * bestAlpha_hat * ZTR).sum(axis=1) but this will produce (F +1, ) instead of (F + 1, 1). My solution is more efficient


    #then we return the optmal solutions
    #the most efficient way is to extract the "original" bestW and bestB from bestW_hat, so revert the initial transformation
    #since bestW_hat = np.array([bestW], [bestB])
    bestW = bestW_hat[0:DTR.shape[0], :] 
    bestB = bestW_hat[-1, :] * K #since K cannot be always = 1 (it's chosen by us), we have to scale the bias


    #now check the duality gap
    def SVM_primalObj(W_hat):
        #compute the primal objective at the solution W_hat
        #first term
        first_term = 0.5 * np.linalg.norm(W_hat)**2
        #second term
        #W_hat = (F + 1, 1) 
        #DTR_hat = (F + 1, N)
        #so it'a matrix vector product
        #vrow(W_hat) @ DTR_hat
        #the result is S = (1, F + 1) * (F + 1, N) = (1, N)
        S = vrow(W_hat) @ DTR_hat       #S = (1, N)

        #then ZTR = (N, ) -> (1, N)
        #in this expression, vrow(ZTR) * S = (1, N) * (1, N) = (1, N)
        #so, 1 and 0 are not scalare, they are broadcasted and are like
        #0 -> np.zeros((1, DTR.shape[1]))
        #1 -> np.ones((1, DTR.shape[1]))
        #np.maximum(array1, array2, ...) is faster than np.amax([array1, array2, ..], axis=0), so use np.maximum
        second_term = np.maximum(0, 1 - vrow(ZTR) * S)
        #sum over all samples and multiply by constant C
        second_term = C * np.sum(second_term)

        return first_term + second_term
    

    #compute duality gap
    primal_optimum = SVM_primalObj(bestW_hat)
    dual_optimum = SVM_dualObj(bestAlpha_hat)
    #in this case, since with the L-BFGS-B we are minimizing the negative of the dual objective function
    #dua_optimum will be negative, so do: dualityGap = primal_optimum + dual_optimum[0]
    dualityGap = primal_optimum +  dual_optimum[0] #do not take the gradient, just the function of the dual
    #print(f"SVM_Linear_SoftMargin, hyperparams C = {C}, K = {K}, bestAlpha_hat (dual) = {bestAlpha_hat}, bestW_hat (primal) = {bestW_hat} bestW (primal) = {bestW}, bestB (primal) = {bestB}\n")
    print(f"SVM_Linear_SoftMargin, hyperparams C = {C}, K = {K}, Primal(bestW_hat) = {primal_optimum}, Dual(bestAlpha_hat) = {-dual_optimum[0]}, computed duality gap: {dualityGap}")


    return bestW, bestB, primal_optimum, dual_optimum[0], dualityGap
    
        


def fit_SVM_Linear_SoftMargin(DTR, LTR, DVAL, LVAL, C, K = 1, appPriorTrue=0.5):
    """
    Train and fit the SVM Linear Soft Margin classifier
    """

    bestW, bestB, primal_loss, optimal_loss, dualitygap = train_SVM_Linear_SoftMargin_Dual(DTR, LTR, C, K)

    #then, compute the score
    #compute s = bestW^T * DVAL + bestB
    #bestW = (F, 1) -> bestW^T = (1, F)
    #DVAL = (F, N)
    #bestB = scalar (bias)
    #scores = (1, F) * (F, N) + scalar = (1, N)
    #it's better to ravel the scores, so scores = (N, )
    #since the dcf functions expect a raveled vector
    scores = (vrow(bestW) @ DVAL).ravel() + bestB

    #decision rule: assign class H_T if score > 0, otherwise assign class H_F
    #DVAL = (F, N)
    #score = (F, N)
    #just doing PVAL = scores > 0 returns a vector of [True, False, True,....]
    #so, since we know that True * 1 = 1; False * 1 = 0 (True and False are casted to 1, 0 if we do the multiplication)
    PVAL = (scores > 0) * 1

    #calculate error rate
    errorRate = np.mean(LVAL != PVAL)

    #calculate DCF, min DCF using appPriorTrue
    minDCF = computeMinEmpiricalBayesRisk_Normalized(scores, LVAL, appPriorTrue, 1.0, 1.0)
    DCF = computeEmpiricalBayesRisk_Normalized(scores, LVAL, appPriorTrue, 1.0, 1.0)

    return bestW, bestB, minDCF, DCF, errorRate, primal_loss, optimal_loss, dualitygap