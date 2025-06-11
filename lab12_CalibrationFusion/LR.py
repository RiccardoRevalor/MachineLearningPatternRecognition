# LOGISTIC REGRESSION MODEL
import numpy as np
import scipy.optimize as opt                    #for fmin_l_bfgs_b
import matplotlib.pyplot as plt                 #for plotting
from mean_covariance import vcol, vrow          #for vector column and row
from decisionModelsEvaluation import computeEmpiricalBayesRisk_Normalized, computeMinEmpiricalBayesRisk_Normalized  #for DCF, minDCF



#1. NON PRIOR WEIGHTED LOGISTIC REGRESSION - TRAIN
def trainLogReg(DTR, LTR, l, manual_grad=True): 
    """
    Train a logistic regression classifier using LTR as labels and DTR as data.
    l is the regularization parameter (lambda).
    Params:
    - DTR: numpy array of shape (n_features, n_samples)
    - LTR: numpy array of shape (n_samples,)
    - l: float, the regularization parameter
    - manual_grad: boolean, if True, the gradient is computed manually, otherwise it is computed using approx_grad
    """

    #LTR: ACTUAL labels -> they the c_i
    #compute z_i = 2 * c_i - 1 -> ZTR = 2 * LTR - 1
    #z_i is used in the Logistic Loss, not c_i
    ZTR = 2 * LTR - 1


    def logreg_obj(v):
        """
        Compute the Objective function for logistic regression.
        v is the vector of parameters in the form g. w, b = v[0:-1], v[-1]
        Parameter:
        - v: numpy array of shape (n_features + 1,)
        Returns:
        - f: float, the value of the objective function
        """
        #extract w and b from v
        w = v[:-1]  #weights
        b = v[-1]   #bias   

        #Now, the objective function is the Logistic Loss which is:
        #f(w, b) = 0.5 * l * ||w||^2 + 1/n * sum_i=1^n log(1 + exp(-z_i * (w^T x_i + b)))
        #so it's the sum of two terms:
        #1. the regularization term: 0.5 * l * ||w||^2
        #2. the average logistic loss term: 1/n * sum_i=1^n log(1 + exp(-z_i * (w^T x_i + b)))


        #compute regularization term (= norm penality)
        normPenalty = 0.5 * l * np.linalg.norm(w)**2

        #to compute the term: log (1 + exp(-z_i * (w^T x_i + b)))
        #we can exploit numpy broadcasting + logaddexp
        #so first we build a vector of scores S = [(w^T x1 + b). . .(w^T xn + b)]
        #then we reshape S to a 1-D array of shape (n_samples, 1)
        S = (vcol(w).T @ DTR).ravel() + b

        #then we exploit broadcasting to compute -z_i * (w^T x_i + b) -> in code it's -ZTR * S -> this term is te full exponent
        exponent = -ZTR * S

        #then we exploit logaddexp: since the log (1 + exp(-z_i * (w^T x_i + b))) can lead to numerical issues, use logaddexp
        #logaddexp(a, b) = log(exp(a) + exp(b))
        #logaddexp(0, exponent) = log(1 + exp(exponent)) -> this is all the second term of the objective function, we just need to compute the mean of it
        logTerm = np.logaddexp(0, exponent)
        avgLogTerms = logTerm.mean()


        #if manual_grad is True, we compute the gradient manually
        if (manual_grad):
            #compute the vector G of the deerivatives of the log of the (1 + exp(-z_i * (w^T x_i + b)))
            G = -ZTR / (1.0 + np.exp(ZTR * S))
            #compute G_i * x_i with broadcasting
            Gixi = vrow(G) * DTR
            #compute derivatives of the objective function wrt to w: dobjF/dw
            der_w = l * w.ravel() + Gixi.mean(axis=1) #mean over the n samples
            #compute drivative of the objective function wrt to b: dobjF/db
            der_b = G.mean() #mean over the n samples

            #pack the two derivatives in a single array
            #stack horizontally the array der_w and the scalar
            #np.hstack is safer than np.array since it will not raise an error if the two arrays have different shapes
            v_grad = np.hstack((der_w, der_b))

            #return the value of the objective function and the gradient
            return normPenalty + avgLogTerms, v_grad

        #if manual_grad is False, we return only the value of the objective function
        return normPenalty + avgLogTerms
    
    #so, the outer function has to invoke the scipy optimizer (fmin_l_bfgs_b) passing the function logreg_obj and the initial x0 which is a numpy array of all zeros and approx_grad = True
    #in this version, we will not manually compute the gradient of the function, but we will use the approx_grad = True option of fmin_l_bfgs_b
    #approx_grad = it depends on the value of manual_grad, if manual_grad is True, we will compute the gradient manually, otherwise we will use the approx_grad = True option of fmin_l_bfgs_b
    xf = opt.fmin_l_bfgs_b(func = logreg_obj, x0 = np.zeros(DTR.shape[0]+1), approx_grad=not manual_grad)
    
    #xf is a tuple with the first element being the minimum point of the function f, the second element being the value of the function f at the minimum point xf[0], and the third element being a dictionary with information about the optimization process
    
    #extract w_min, b_min
    w_min = xf[0][:-1] #weights which minimize the objective function
    b_min = xf[0][-1]  #bias which minimizes the objective function

    #extract value of objective function in (w_min, b_min)
    objMin = xf[1] 
    
    return w_min, b_min, objMin

#2. NON PRIOR WEIGHTED LOGISTIC REGRESSION - PREDICT
def fitLogReg(DTR, LTR, DVAL, LVAL, lambdas, PriorTrue = 0.5, manual_grad=True):
    """
    Train a logistic regression classifier using LTR as labels and DTR as data.
    lambdas is a list of regularization parameters (lambda).
    Parameters:
    - DTR: numpy array of shape (n_features, n_samples), training data
    - LTR: numpy array of shape (n_samples,), training labels
    - DVAL: numpy array of shape (n_features, n_samples), validation data
    - LVAL: numpy array of shape (n_samples,), validation labels
    - lambdas: list of regularization parameters (lambda)
    - PriorTrue: float, the application prior probability of the positive class -> in this case it's not prior weighted, we just subtract the empirical prior log odds from the scores and then threshold the result with the application prior log odds
    - manual_grad: boolean, if True, the gradient is computed manually, speeding up the computations, otherwise it is computed using approx_grad
    """

    #STEP 1: RETRIEVE MODEL PARAMS (w, b)
    #retrieve the parameters which minimize the objective function
    #change lambda from 10^(-3) to 1.0, by incrementing it by a decimal order every time
    parameters_l = {} #key: lambda, value: (w_min, b_min)
    objMin_l = {} #key: lambda, value: (objMin)
    for l in lambdas:
        w_min, b_min, objMin = trainLogReg(DTR, LTR, l, manual_grad=manual_grad)
        parameters_l[l] = (w_min, b_min)
        objMin_l[l] = objMin


    #STEP 2: LOG POSTERIORS
    #compute log posteriors ratios using DVAL samples
    scores_l = {} #key: lambda, value: (log posterior ratio for each sample in DVAL)
    for l in lambdas:
        p = parameters_l[l]
        w, b = p[0], p[1]
        S = (vcol(w).T @ DVAL).ravel() + b
        scores_l[l] = S


    #STEP3: DECISION RULE -> PERFORM CLASS ASSIGNMENTS
    #Perform class assignments
    LP_l = {} #key: lambda, value: (log posterior ratio for each sample in DVAL)
    for l in lambdas:
        score = scores_l[l]
        LP = np.zeros(score.shape) #predicted labels array
        LP[score > 0] = 1 #assign label 1 to samples with score > 0
        LP[score < 0] = 0 #assign label 0 to samples with score < 0
        LP_l[l] = LP #store the predicted labels for each lambda

    #compute error rates 
    err_l = {} #key: lambda, value: error rate
    for l in lambdas:
        LP = LP_l[l]
        err = (LP != LVAL).sum() / float(LVAL.size) * 100
        err_l[l] = err #store the error rate for each lambda



    #STEP4: COMPUTE LLR LIKE SCORES BY SUBTRCACTING EMPIRICAL PRIOR LOG ODDS FROM THE SCORES
    #Empirical priors computation from DTR
    pi_emp_h1 = np.sum(LTR == 1) / LTR.size
    pi_emp_h0 = np.sum(LTR == 0) / LTR.size
    llr_like_scores_l = {} #key: lambda, value: (s_llr)
    for l in lambdas:
        score = scores_l[l]
        #subtract empirical prior log odds
        s_llr = score - np.log(pi_emp_h1 / pi_emp_h0) #the same as s_llr = score - np.log(pi_emp_h1 / (1- pi_emp_h1))
        llr_like_scores_l[l] = s_llr #store the log likelihood ratio scores for each lambda




    #STEP5: COMPUTE DDCF, MIN DCF
    #Compute dcf, min_dcf
    dcf_l = {} #key: lambda, value: (dcf)
    min_dcf_l = {} #key: lambda, value: (min_dcf)
    for l in lambdas:
        dcf = computeEmpiricalBayesRisk_Normalized(llr_like_scores_l[l], LVAL, PriorTrue, 1.0, 1.0)
        min_dcf = computeMinEmpiricalBayesRisk_Normalized(llr_like_scores_l[l], LVAL, PriorTrue, 1.0, 1.0)
        dcf_l[l] = dcf #store the dcf for each lambda
        min_dcf_l[l] = min_dcf #store the min_dcf for each lambda



    #STEP6: TABLE
    # Create the table using matplotlib
    lambdas_table = sorted(lambdas)
    error_rates = [err_l[l] for l in lambdas_table]
    min_dcfs = [min_dcf_l[l] for l in lambdas_table]
    dcfs = [dcf_l[l] for l in lambdas_table]
    objMin_list = [objMin_l[l] for l in lambdas_table]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Create table data
    table_data = [
        [f"{l:.5f}", f"{objMin:.4f}", f"{err:.1f}%", f"{min_dcf:.4f}", f"{dcf:.4f}"]
        for l, objMin, err, min_dcf, dcf in zip(lambdas_table, objMin_list, error_rates, min_dcfs, dcfs)
    ]

    # Create the table
    table = ax.table(cellText=table_data,
                     colLabels=["$\lambda$", "$\mathcal{J}(\mathbf{w}^*, b^*)$", "Error rate", f"minDCF ($\pi_T = {PriorTrue}$)", f"actDCF ($\pi_T = {PriorTrue}$)"],
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title("Logistic Regression Performance", pad=20)
    plt.tight_layout()
    plt.show()


    #return the parameters, error rates, min_dcf, dcf
    return parameters_l, err_l, min_dcf_l, dcf_l, objMin_l, llr_like_scores_l


#---------------------------------------------------------------------------------------------------------------------------------


#3. PRIOR WEIGHTED LOGISTIC REGRESSION - TRAIN
def trainLogReg_PriorWeighted(DTR, LTR, l, manual_grad=True, PriorTrue=0.5): 
    """
    Train a logistic regression classifier using LTR as labels and DTR as data.
    l is the regularization parameter (lambda).
    Params:
    - DTR: numpy array of shape (n_features, n_samples)
    - LTR: numpy array of shape (n_samples,)
    - l: float, the regularization parameter
    - manual_grad: boolean, if True, the gradient is computed manually, otherwise it is computed using approx_grad
    - PriorTrue: float, the application prior probability of the positive class 
    """

    #LTR: ACTUAL labels -> they the c_i
    #compute z_i = 2 * c_i - 1 -> ZTR = 2 * LTR - 1
    #z_i is used in the Logistic Loss, not c_i
    ZTR = 2 * LTR - 1

    #compute weights for the 2 classes using the application prior
    nT = np.sum(LTR == 1) 
    weightTrue = PriorTrue / nT
    nF = LTR.size - nT
    weightFalse = (1 - PriorTrue) / nF


    def logreg_obj(v):
        """
        Compute the Objective function for logistic regression.
        v is the vector of parameters in the form g. w, b = v[0:-1], v[-1]
        Parameter:
        - v: numpy array of shape (n_features + 1,)
        Returns:
        - f: float, the value of the objective function
        """
        #extract w and b from v
        w = v[:-1]  #weights
        b = v[-1]   #bias   

        #Now, the objective function is the Logistic Loss which is:
        #f(w, b) = 0.5 * l * ||w||^2 +  sum_i=1^n log(1 + exp(-z_i * (w^T x_i + b)))
        #so it's the sum of two terms:
        #1. the regularization term: 0.5 * l * ||w||^2
        #2. sum of weighted logistic loss term: sum_i=1^n weight_i log(1 + exp(-z_i * (w^T x_i + b)))


        #compute regularization term (= norm penality)
        normPenalty = 0.5 * l * np.linalg.norm(w)**2

        #to compute the term: log (1 + exp(-z_i * (w^T x_i + b)))
        #we can exploit numpy broadcasting + logaddexp
        #so first we build a vector of scores S = [(w^T x1 + b). . .(w^T xn + b)]
        #then we reshape S to a 1-D array of shape (n_samples, 1)
        S = (vcol(w).T @ DTR).ravel() + b

        #then we exploit broadcasting to compute -z_i * (w^T x_i + b) -> in code it's -ZTR * S -> this term is te full exponent
        exponent = -ZTR * S

        #then we exploit logaddexp: since the log (1 + exp(-z_i * (w^T x_i + b))) can lead to numerical issues, use logaddexp
        #logaddexp(a, b) = log(exp(a) + exp(b))
        logTerm = np.logaddexp(0, exponent)
        #multiply the log terms by the weights, depending on wether z_i (= ZTR in the code) is 1 or -1
        logTerm[ZTR == 1] *= weightTrue
        logTerm[ZTR == -1] *= weightFalse
        #then sum all the loss contributions by each sampes
        logTermSum = np.sum(logTerm)



        #if manual_grad is True, we compute the gradient manually
        if (manual_grad):
            #compute the vector G of the deerivatives of the log of the (1 + exp(-z_i * (w^T x_i + b)))
            G = -ZTR / (1.0 + np.exp(ZTR * S))
            
            #multiply the G vector by the weights, depending on wether z_i (= ZTR in the code) is 1 or -1
            G[ZTR == 1] *= weightTrue
            G[ZTR == -1] *= weightFalse

            #compute G_i * x_i with broadcasting
            Gixi = vrow(G) * DTR
            #compute derivatives of the objective function wrt to w: dobjF/dw
            der_w = l * w.ravel() + Gixi.sum(axis=1) #sum all n samples

            #compute drivative of the objective function wrt to b: dobjF/db
            der_b = G.sum() #sum all the n samples

            #pack the two derivatives in a single array
            #stack horizontally the array der_w and the scalar
            #np.hstack is safer than np.array since it will not raise an error if the two arrays have different shapes
            v_grad = np.hstack((der_w, der_b))

            #return the value of the objective function and the gradient
            return normPenalty + logTermSum, v_grad

        #if manual_grad is False, we return only the value of the objective function
        return normPenalty + logTermSum
    
    #so, the outer function has to invoke the scipy optimizer (fmin_l_bfgs_b) passing the function logreg_obj and the initial x0 which is a numpy array of all zeros and approx_grad = True
    #in this version, we will not manually compute the gradient of the function, but we will use the approx_grad = True option of fmin_l_bfgs_b
    #approx_grad = it depends on the value of manual_grad, if manual_grad is True, we will compute the gradient manually, otherwise we will use the approx_grad = True option of fmin_l_bfgs_b
    xf = opt.fmin_l_bfgs_b(func = logreg_obj, x0 = np.zeros(DTR.shape[0]+1), approx_grad=not manual_grad)
    
    #xf is a tuple with the first element being the minimum point of the function f, the second element being the value of the function f at the minimum point xf[0], and the third element being a dictionary with information about the optimization process
    
    #extract w_min, b_min
    w_min = xf[0][:-1] #weights which minimize the objective function
    b_min = xf[0][-1]  #bias which minimizes the objective function

    #extract value of objective function in (w_min, b_min)
    objMin = xf[1] 
    
    return w_min, b_min, objMin



#4. PRIOR WEIGHTED LOGISTIC REGRESSION - PREDICT
def fitLogReg_PriorWeighted(DTR, LTR, DVAL, LVAL, lambdas, PriorTrue=0.5, manual_grad=True):
    """
    Train a logistic regression classifier using LTR as labels and DTR as data.
    lambdas is a list of regularization parameters (lambda).
    Parameters:
    - DTR: numpy array of shape (n_features, n_samples), training data
    - LTR: numpy array of shape (n_samples,), training labels
    - DVAL: numpy array of shape (n_features, n_samples), validation data
    - LVAL: numpy array of shape (n_samples,), validation labels
    - lambdas: list of regularization parameters (lambda)
    - PriorTrue: float, the application prior probability of the positive class 
    """

    #STEP 1: RETRIEVE MODEL PARAMS (w, b)
    #retrieve the parameters which minimize the objective function
    #change lambda from 10^(-3) to 1.0, by incrementing it by a decimal order every time
    parameters_l = {} #key: lambda, value: (w_min, b_min)
    objMin_l = {} #key: lambda, value: (objMin)
    for l in lambdas:
        w_min, b_min, objMin = trainLogReg_PriorWeighted(DTR, LTR, l, manual_grad, PriorTrue)
        parameters_l[l] = (w_min, b_min)
        objMin_l[l] = objMin
        


    #STEP 2: LOG POSTERIORS
    #compute log posteriors ratios using DVAL samples
    scores_l = {} #key: lambda, value: (log posterior ratio for each sample in DVAL)
    for l in lambdas:
        p = parameters_l[l]
        w, b = p[0], p[1]
        S = (vcol(w).T @ DVAL).ravel() + b
        scores_l[l] = S


    #STEP3: DECISION RULE -> PERFORM CLASS ASSIGNMENTS
    #Perform class assignments
    LP_l = {} #key: lambda, value: (log posterior ratio for each sample in DVAL)
    for l in lambdas:
        score = scores_l[l]
        LP = np.zeros(score.shape) #predicted labels array
        LP[score > 0] = 1 #assign label 1 to samples with score > 0
        LP[score < 0] = 0 #assign label 0 to samples with score < 0
        LP_l[l] = LP #store the predicted labels for each lambda

    #compute error rates 
    err_l = {} #key: lambda, value: error rate
    for l in lambdas:
        LP = LP_l[l]
        err = (LP != LVAL).sum() / float(LVAL.size) * 100
        err_l[l] = err #store the error rate for each lambda



    #STEP4: COMPUTE LLR LIKE SCORES BY SUBTRCACTING APPLICATION PRIOR LOG ODDS FROM THE SCORES
    #Empirical priors computation from DTR
    llr_like_scores_l = {} #key: lambda, value: (s_llr)
    for l in lambdas:
        score = scores_l[l]
        #subtract application prior log odds
        s_llr = score - np.log(PriorTrue / (1- PriorTrue)) #the same as s_llr = score - np.log(pi_emp_h1 / (1- pi_emp_h1))
        llr_like_scores_l[l] = s_llr #store the log likelihood ratio scores for each lambda




    #STEP5: COMPUTE DDCF, MIN DCF
    #Compute dcf, min_dcf
    dcf_l = {} #key: lambda, value: (dcf)
    min_dcf_l = {} #key: lambda, value: (min_dcf)
    for l in lambdas:
        dcf = computeEmpiricalBayesRisk_Normalized(llr_like_scores_l[l], LVAL, PriorTrue, 1.0, 1.0)
        min_dcf = computeMinEmpiricalBayesRisk_Normalized(llr_like_scores_l[l], LVAL, PriorTrue, 1.0, 1.0)
        dcf_l[l] = dcf #store the dcf for each lambda
        min_dcf_l[l] = min_dcf #store the min_dcf for each lambda



    #STEP6: TABLE
    # Create the table using matplotlib
    lambdas_table = sorted(lambdas)
    error_rates = [err_l[l] for l in lambdas_table]
    min_dcfs = [min_dcf_l[l] for l in lambdas_table]
    dcfs = [dcf_l[l] for l in lambdas_table]
    objMin_list = [objMin_l[l] for l in lambdas_table]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Create table data
    table_data = [
        [f"{l:.5f}", f"{objMin:.4f}", f"{err:.1f}%", f"{min_dcf:.4f}", f"{dcf:.4f}"]
        for l, objMin, err, min_dcf, dcf in zip(lambdas_table, objMin_list, error_rates, min_dcfs, dcfs)
    ]

    # Create the table
    table = ax.table(cellText=table_data,
                     colLabels=["$\lambda$", "$\mathcal{J}(\mathbf{w}^*, b^*)$", "Error rate", "minDCF ($\pi_T = 0.5$)", f"actDCF ($\pi_T = {PriorTrue}$)"],
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title("Logistic Regression Performance", pad=20)
    plt.tight_layout()
    plt.show()

    #return the parameters, error rates, min_dcf, dcf
    return parameters_l, err_l, min_dcf_l, dcf_l, objMin_l, llr_like_scores_l