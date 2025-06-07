# DECISION MODELS EVALUATION
import numpy as np
import matplotlib.pyplot as plt


# function to compute the confusion matrix
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

        
        #add classCol to the confusion matrix in a row major fashion
        ConfMatrix[classPredicted, :] = classRow


    return ConfMatrix



# function to compute the confusion matrix from the log likelihoods and priors
# always import MVG before using this function!
def computeConfMatrixFromLL(LVAL, logLikelihoods, Priors, MVG, useLog=True):
    """
    Compute the confusion matrix for the predicted labels and the actual labels.
    Args:
    - logLikelihoods: matriix of log likelihoods for each class
    - Priors: array of priors for each class, priors are application dependent
    - MVG: the MVG class object used to compute the joint densities and posteriors
    - useLog: if True, use log likelihoods, else use normal likelihoods

    Returns:
    - Confusion matrix
    """

    SJoint = MVG.computeSJoint(logLikelihoods, Priors, useLog=useLog) #compute the joint densities by multiplying the score matrix S with the Priors
    SPost = MVG.computePosteriors(SJoint, useLog=True)  #compute the posteriors by normalizing the joint densities
    PVAL = np.argmax(SPost, axis=0) #select the class with the highest posterior probability for each sample, set axis=0 to select the class with the highest posterior probability for each sample

    #call the computeConfMatrix function to compute the confusion matrix
    return computeConfMatrix(PVAL, LVAL)



# function to compute the optimal Bayes decision for a given prior and cost function -> BINARY CLASSIFICATION ONLY
# function computeConfMatrix must be imported before using this function!
def optimalBayesDecisionClassifier(llrs, LVAL, PriorTrue, Cfn, Cfp):
    """
    Compute the optimal Bayes decision for a given prior and cost function.
    And perform classification obtaining a confusion matrix.
    Args:
    - llrs: log likelihood ratios
    - LVAL: actual labels
    - PriorTrue: Prior probability of the true class
    - Cfn: Cost of false negative
    - Cfp: Cost of false positive
    Returns:
    - Confusion matrix
    - Optimal decision threshold
    """

    #compute optimal threshold
    t = -1 * np.log((PriorTrue * Cfn) / ((1 - PriorTrue) * Cfp))


    #Classification rule: if llr > t, classify as 1, else classify as 0
    PVAL = np.where(llrs > t, 1, 0)


    #compute confusion matrix
    return computeConfMatrix(PVAL, LVAL), t


# function to compute the empirical Bayes risk for a given prior and cost function -> BINARY CLASSIFICATION ONLY
def computeEmpiricalBayesRisk(llrs, LVAL, PriorTrue, Cfn, Cfp):
    """
    Compute the empirical Bayes risk for a given prior and cost function.
    Args:
    - llrs: log likelihood ratios
    - LVAL: actual labels
    - PriorTrue: Prior probability of the true class
    - Cfn: Cost of false negative
    - Cfp: Cost of false positive
    Returns:
    - Empirical Bayes risk
    """

    confMatrix, _ = optimalBayesDecisionClassifier(llrs, LVAL, PriorTrue, Cfn, Cfp)
    #now, extract the TP, TN, FP and FN from the confusion matrix
    TP = confMatrix[1, 1] #True Positives
    TN = confMatrix[0, 0] #True Negatives
    FP = confMatrix[1, 0] #False Positives
    FN = confMatrix[0, 1] #False Negatives

    #comupte Pfn, Pfp
    Pfn = FN / (FN + TP) #False Negative Rate
    Pfp = FP / (FP + TN) #False Positive Rate

    return (PriorTrue * Cfn * Pfn) + ((1 - PriorTrue) * Cfp * Pfp) #Empirical Bayes Risk



# function to compute the normalized empirical Bayes risk for a given prior and cost function -> BINARY CLASSIFICATION ONLY
def computeEmpiricalBayesRisk_Normalized(llrs, LVAL, PriorTrue, Cfn, Cfp):
    """
    Compute the empirical Bayes risk for a given prior and cost function.
    Args:
    - llrs: log likelihood ratios
    - LVAL: actual labels
    - PriorTrue: Prior probability of the true class
    - Cfn: Cost of false negative
    - Cfp: Cost of false positive
    Returns:
    - Normalized Empirical Bayes risk
    """

    confMatrix, _ = optimalBayesDecisionClassifier(llrs, LVAL, PriorTrue, Cfn, Cfp)
    #now, extract the TP, TN, FP and FN from the confusion matrix
    TP = confMatrix[1, 1] #True Positives
    TN = confMatrix[0, 0] #True Negatives
    FP = confMatrix[1, 0] #False Positives
    FN = confMatrix[0, 1] #False Negatives

    #comupte Pfn, Pfp
    Pfn = FN / (FN + TP) #False Negative Rate
    Pfp = FP / (FP + TN) #False Positive Rate

    Bemp = (PriorTrue * Cfn * Pfn) + ((1 - PriorTrue) * Cfp * Pfp) #Empirical Bayes Risk


    #Now calculate the Bemp for the two dummy systems:
    Bemp_dummy1= PriorTrue * Cfn
    Bemp_dummy2 = (1 - PriorTrue) * Cfp
    #take the min between the two Bemp dummy
    Bemp_dummy = min(Bemp_dummy1, Bemp_dummy2)

    return Bemp / Bemp_dummy #Normalized Empirical Bayes Risk



# function to compute the minimum Bayes normalized empirical risk for a given prior and cost function -> BINARY CLASSIFICATION ONLY
def computeMinEmpiricalBayesRisk_Normalized(scores, LVAL, PriorTrue, Cfn, Cfp):
    """
    Compute the minimum Bayes normalized empirical risk (= the min DCF) for a given prior and cost function.
    Args:
    - scores: log likelihood ratios
    - LVAL: actual labels
    - PriorTrue: Prior probability of the true class
    - Cfn: Cost of false negative
    - Cfp: Cost of false positive
    Returns:
    - Minimum Bayes normalized empirical risk
    """


    #1. sort the scores in increasing order to use them as thresholds
    sortedScores = np.sort(scores)

    #2. add the -inf and +inf to the sorted scores to use them as thresholds
    sortedScores = np.concatenate(([-np.inf], sortedScores, [np.inf])) 
    #now we have sortedScores = (-inf, s0, s1, s2, ..., sN, +inf)

    #3. for each threshold, compute the empirical Bayes risk
    DCFList = [] #initialize the DCFList with an empty list

    for t in sortedScores:
        #score IS THE THRESHOLD!
        #Classification rule: if llr > t, classify as 1, else classify as 0
        PVAL = np.where(scores > t, 1, 0)

        #compute confusion matrix
        confMatrix = computeConfMatrix(PVAL, LVAL)


        #now, extract the TP, TN, FP and FN from the confusion matrix
        TP = confMatrix[1, 1] #True Positives
        TN = confMatrix[0, 0] #True Negatives
        FP = confMatrix[1, 0] #False Positives
        FN = confMatrix[0, 1] #False Negatives

        #comupte Pfn, Pfp
        Pfn = FN / (FN + TP) #False Negative Rate
        Pfp = FP / (FP + TN) #False Positive Rate

        Bemp = (PriorTrue * Cfn * Pfn) + ((1 - PriorTrue) * Cfp * Pfp) #Empirical Bayes Risk


        #Now calculate the Bemp for the two dummy systems:
        Bemp_dummy1= PriorTrue * Cfn
        Bemp_dummy2 = (1 - PriorTrue) * Cfp
        #take the min between the two Bemp dummy
        Bemp_dummy = min(Bemp_dummy1, Bemp_dummy2)

        DCF_i = Bemp / Bemp_dummy #Normalized Empirical Bayes Risk

        DCFList.append(DCF_i) #append the DCF_i to the DCFList


    #4. find the minimum DCF in the DCFArray
    return min(DCFList) 


# function to plot the ROC curve -> BINARY CLASSIFICATION ONLY
def plotROC_FPRvsTPR(scores, LVAL):
    """
    Plot the ROC curve for a given set of scores and labels.
    Args:
    - scores: log likelihood ratios
    - LVAL: actual labels
    """


    #1. sort the scores in increasing order to use them as thresholds
    sortedScores = np.sort(scores)

    #2. add the -inf and +inf to the sorted scores to use them as thresholds
    sortedScores = np.concatenate(([-np.inf], sortedScores, [np.inf])) 
    #now we have sortedScores = (-inf, s0, s1, s2, ..., sN, +inf)

    #initialize the x and y axis for the ROC curve
    xAxis = [] 
    yAxis = [] 

    for t in sortedScores:
        #score IS THE THRESHOLD!
        #Classification rule: if llr > t, classify as 1, else classify as 0
        PVAL = np.where(scores > t, 1, 0)

        #compute confusion matrix
        confMatrix = computeConfMatrix(PVAL, LVAL)


        #now, extract the TP, TN, FP and FN from the confusion matrix
        TP = confMatrix[1, 1] #True Positives
        TN = confMatrix[0, 0] #True Negatives
        FP = confMatrix[1, 0] #False Positives
        FN = confMatrix[0, 1] #False Negatives

        #comupte Pfn, Pfp
        Pfn = FN / (FN + TP) #False Negative Rate
        Pfp = FP / (FP + TN) #False Positive Rate
        Ptp = 1 - Pfn #True Positive Rate

        #ROC: Pfp vs Ptp
        #Pfp on x axis, Ptp on y axis
        xAxis.append(Pfp)
        yAxis.append(Ptp)

    #3. plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', markersize=1, color='blue')
    plt.title("ROC Curve: FPR vs TPR", pad=20, fontsize=14)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.grid(True)
    #limit the x and y axis to [0, 1], since the values of FPR and TPR are between 0 and 1
    #this is done for better visualization of the ROC curve, not to have empty parts of the plot
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


# function to plot the Bayes error plots for a given range of log odds ratios and scores -> BINARY CLASSIFICATION ONLY
def plotBayesErrorPlots(effPriorLogOdds, scores, LVAL, title="Bayes Error Plots: DCF and min DCF vs Effective Prior Log Odds", xticks=31):
    """
    Plot the Bayes error plots for a given range of log odds ratios and scores.
    Args:
    - logOddsRange: range of log odds ratios -> used to compute the effective Prior
    - scores: log likelihood ratios
    - LVAL: actual labels
    - title: title of the plot
    """

    #effPriorLogOdds will be the x axis of the plot
    xAxis = effPriorLogOdds
    #the plot will have two series: one for the DCF and the other for the min DCF -> they will be on the y axis
    #(ofc the are *normalized*)
    series0_yAxis = [] #y axis for the DCF
    series1_yAxis = [] #y axis for the min DCF

    #Cfp = 1, Cfn = 1

    #computeEmpiricalBayesRisk_Normalized(llrs, LVAL, PriorTrue, Cfn, Cfp):

    for tildeP in effPriorLogOdds:

        #compute the effective Prior from tildeP
        effectivePrior = 1 / (1 + np.exp(-tildeP))

        # compute DCF
        DCF = computeEmpiricalBayesRisk_Normalized(scores, LVAL, effectivePrior, 1, 1)
        series0_yAxis.append(DCF)
        # compute min DCF
        minDCF = computeMinEmpiricalBayesRisk_Normalized(scores, LVAL, effectivePrior, 1, 1)
        series1_yAxis.append(minDCF)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(xAxis, series0_yAxis, marker='o', linestyle='-', markersize=1, label='DCF', color='red')
    plt.plot(xAxis, series1_yAxis, marker='o', linestyle='-', markersize=1, label='min DCF', color='blue')
    plt.title(title, pad=20, fontsize=14)
    plt.xlabel("Effective Prior Log Odds")
    plt.ylim([0, 1.1])
    plt.xlim([min(xAxis), max(xAxis)]) #set the x axis limits to the min and max of the x axis which are the effectivePriorLogOdds
    plt.grid(True)
    plt.legend()

    # Customize x-axis ticks
    num_ticks = xticks  # Number of ticks to display on the x-axis 
    # Generate evenly spaced ticks, rotate them by 45 degrees and align them to the right to make them more clear
    plt.xticks(np.linspace(min(xAxis), max(xAxis), num_ticks), rotation=45, ha='right') 
    plt.show()


def plotMisscalibrationError(effPriorLogOdds, scores, LVAL, title="Misscalibration Error vs Effective Prior Log Odds", xticks=31):
    """
    Plot the misscalibration error for a given range of log odds ratios and scores.
    Args:
    - logOddsRange: range of log odds ratios -> used to compute the effective Prior
    - scores: log likelihood ratios
    - LVAL: actual labels
    - title: title of the plot
    """

    #effPriorLogOdds will be the x axis of the plot
    xAxis = effPriorLogOdds
    #the plot will have two series: one for the DCF and the other for the min DCF -> they will be on the y axis
    #(ofc the are *normalized*)
    series0_yAxis = [] #y axis for the DCF

    #computeEmpiricalBayesRisk_Normalized(llrs, LVAL, PriorTrue, Cfn, Cfp):

    for tildeP in effPriorLogOdds:

        #compute the effective Prior from tildeP
        effectivePrior = 1 / (1 + np.exp(-tildeP))

        # compute DCF
        DCF = computeEmpiricalBayesRisk_Normalized(scores, LVAL, effectivePrior, 1, 1)
        # compute min DCF
        minDCF = computeMinEmpiricalBayesRisk_Normalized(scores, LVAL, effectivePrior, 1, 1)

        series0_yAxis.append(DCF - minDCF) #append the misscalibration error to the y axis

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(xAxis, series0_yAxis, marker='o', linestyle='-', markersize=1, label='Misscalibration Error', color='red')
    plt.title(title, pad=20, fontsize=14)
    plt.xlabel("Effective Prior Log Odds")
    #plt.ylim([0, max(series0_yAxis) + ((max(series0_yAxis) - min(series0_yAxis)) / 4)]) #set the y axis limits to the min and max of the y axis which are the misscalibration error
    plt.xlim([min(xAxis), max(xAxis)]) #set the x axis limits to the min and max of the x axis which are the effectivePriorLogOdds
    plt.grid(True)
    plt.legend()

    # Customize x-axis ticks
    num_ticks = xticks  # Number of ticks to display on the x-axis 
    # Generate evenly spaced ticks, rotate them by 45 degrees and align them to the right to make them more clear
    plt.xticks(np.linspace(min(xAxis), max(xAxis), num_ticks), rotation=45, ha='right') 
    plt.show()


# function to plot the Bayes error plots for different models to compare them-> BINARY CLASSIFICATION ONLY
def plotBayesErrorPlotsMoreModels(models, title = "Bayes Error Plots: DCF and min DCF vs Effective Prior Log Odds", xticks=31):
    """
    Plot the Bayes error plots for a given range of log odds ratios and scores.
    Args:
        - models: list of lists containing:
            - effPriorsLogOdds: range of log odds ratios -> used to compute the effective Prior
            - scores: log likelihood ratios
            - LVAL: actual labels
            - model_name: name of the model for the legend
    """

    plt.figure(figsize=(8, 6))

    for model in models:
        # unpack the parameters for the model
        effPriorLogOdds, scores, LVAL, model_name = model

        # set y axis for the specific model
        series0_yAxis = [] # y axis for the DCF
        series1_yAxis = [] # y axis for the min DCF

        # effPriorLogOdds will be the x axis of the plot
        xAxis = effPriorLogOdds

        for tildeP in effPriorLogOdds:
            # compute the effective Prior from tildeP
            effectivePrior = 1 / (1 + np.exp(-tildeP))
            #print(f"Effective Prior: {effectivePrior}")

            # compute DCF
            DCF = computeEmpiricalBayesRisk_Normalized(scores, LVAL, effectivePrior, 1, 1)
            series0_yAxis.append(DCF)

            # compute min DCF
            minDCF = computeMinEmpiricalBayesRisk_Normalized(scores, LVAL, effectivePrior, 1, 1)
            series1_yAxis.append(minDCF)

        # Plot the results for the current model
        plt.plot(xAxis, series0_yAxis, marker='o', linestyle='-', markersize=1, label=f'DCF ({model_name})')
        plt.plot(xAxis, series1_yAxis, marker='o', linestyle='-', markersize=1, label=f'min DCF ({model_name})')

    plt.title(title, pad=20, fontsize=14)
    plt.xlabel("Effective Prior Log Odds")
    plt.ylim([0, 1.1])
    plt.xlim([min(xAxis), max(xAxis)]) # set the x axis limits to the min and max of the x axis
    plt.grid(True)
    plt.legend()

    # Customize x-axis ticks
    num_ticks = xticks # Number of ticks to display on the x-axis
    # Generate evenly spaced ticks, rotate them by 45 degrees and align them to the right
    plt.xticks(np.linspace(min(xAxis), max(xAxis), num_ticks), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()




###############################################################################################################################################################

# THESE ARE FOR MULTICLASS PROBLEMS
# function to compute the optimal Bayes decision for a given prior and cost function -> MULTI CLASS CLASSIFICATION ONLY
def optimalBayesDecisionClassifier_MultiClass(ll, LVAL, PriorsVect, CostMatrix, MVG):
    """
    Compute the optimal Bayes decision for a given prior and cost function.
    And perform classification obtaining a confusion matrix.
    Args:
    - ll: log likelihood ratios
    - LVAL: actual labels
    - PriorsVect: Prior probability of each class
    - CostMatrix: Cost matrix for each class
    - MVG: the MVG class object used to compute the joint densities and posteriors
    Returns
    - PVAL
    """

    #Compute the matrix of the Posterior probabilities for the 3 class problem

    #so first compute the joint from the lieklihoods and the priors
    SJoint = MVG.computeSJoint(ll, PriorsVect, useLog=True) #compute the joint densities by multiplying the score matrix S with the Priors

    #then compute the posteriors by normalizing the joint densities
    SPost = MVG.computePosteriors(SJoint, useLog=True) #compute the posteriors by normalizing the joint densities
    
    #exponentiante SPost
    SPost = np.exp(SPost) #exponentiate the posteriors to get the probabilities

    #compute matrix of expected Bayes costs
    C = CostMatrix @ SPost 

    #print(C[:, 1]) 

    #take optimal COSTS where the expected Bayes costs are minimum
    PVAL = np.argmin(C, axis=0) #select the class with the lowest expected cost for each sample, set axis=0 to select the class with the lowest expected cost for each sample

    return PVAL #return the optimal classes


# function to compute the NORMALIZED empirical Bayes risk for a given prior and cost function -> MULTI CLASS CLASSIFICATION ONLY
def computeEmpiricalBayesRisk_Normalized_MultiClass(ll, LVAL, PriorVect, CostMatrix):
    """
    Compute the empirical Bayes risk for a given prior and cost function. Valid for multi-class problems.
    Args:
    - ll: log likelihood ratios
    - LVAL: actual labels
    - PriorVect: Prior probability of each class
    - CostMatrix: Cost matrix for each class
    Returns:
    - Confusion Matrix
    - Unnormalized Empirical Bayes risk (unnormalized DCF)
    - Normalized Empirical Bayes risk   (normalized DCF)
    """

    numClasses = len(PriorVect) #number of classes

    PVAL = optimalBayesDecisionClassifier_MultiClass(ll, LVAL, PriorVect, CostMatrix) #these are the predicted classes
    
    #compute confusion matrix
    confMatrix = computeConfMatrix(PVAL, LVAL) #compute the confusion matrix


    #compute the missclassification ratio for each class c
    #it's found by diving M_{i,j} by the sum of all the elements of the j-th column of the confusion matrix
    #this is done for each class c

    R = np.zeros((numClasses, numClasses)) #initialize the missclassification ratio matrix with zeros

    for i in range(numClasses):
        for j in range(numClasses):
            M_ij = confMatrix[i, j] #get the value of the confusion matrix for class i and j
            denominator = np.sum(confMatrix[:, j]) #sum of all the elements of the j-th column of the confusion matrix
            R[i, j] = M_ij / denominator    


    #compute the empirical Bayes risk, so the unnormalized DCF
    innerSumResults = np.zeros(numClasses) #innserSumResults is a vector of zeros with the same size as the number of classes
    for j in range(numClasses):
        #this like computing R[i, j] * CostMatrix[i, j] for each i and then summing all together 
        innerSumResults[j] = np.sum(R[:, j] * CostMatrix[:, j]) #compute the inner sum for each class j

    #Then multiply this the inner Sums of each j by the Prior probability of the class j
    Bemp = np.sum(innerSumResults * PriorVect) #compute the empirical Bayes risk



    #Now that I've got Bemp I have to normalize it by dividing it by the minimum of the two dummy systems:
    Bemp_dummy = np.min(CostMatrix @ PriorVect) #this is the minimum of the two dummy systems

    return confMatrix, Bemp, Bemp / Bemp_dummy
        
         

    
    

    
    
    





    
    