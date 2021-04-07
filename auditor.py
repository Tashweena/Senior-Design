import numpy as np

def auditor(l,l_hat, alpha, delta, X_b, y_b,n,verbose=False):
  """ The goal of the auditor is to check for alpha consistency
    violations for a subgroup, that is it checks whether the predicted mean 
    of the subset deviates too much from the true mean.
    
    Inputs: 
    l - True label function
    l_hat - Predictor
    alpha - hyperparameter threshold for the auditor
    delta - hyperparameter
    X_b, y_b - Datapoints in a subgroup to be audited

    For mean_multicalibration l_hat is the prediction for each point and l=y_b

    Outputs: returns lbda
             lbda = +1 if predicted mean is overestimated
                    -1 if underestimated 
                    else None
  """
  lbda = None
  n_b = len(y_b)
  if n_b > 0:
    y_pred_avg = np.average(l_hat)
    y_true_avg = np.average(y_b) ##might be different with higher moments
    difference = y_pred_avg - y_true_avg
    adj_difference = abs(difference)  - 2 * np.sqrt(np.log(2.0/delta)/(2.0*n_b))    
    denominator = ((n_b / n) - np.sqrt(np.log(2.0/delta)/(2.0*n)))
    
    if denominator < alpha:
      # subgroup too small
      return lbda

    # consistency check
    if (adj_difference >= alpha / denominator):
      lbda = np.sign(difference)
    
    if verbose == True:
      print("Size of subgroup(n_b) is ", n_b, " Batch size(n) is ", n, " Proportion is ", (n_b / n))
      print('Difference between True and Predicted ', difference)
      print("Adjusted_difference is ", adj_difference)
      print("LHS square root term ", 2 * np.sqrt(np.log(2.0/delta)/(2.0*n_b)))
      print("RHS term (alpha/denominator) is ", alpha / ((n_b / n) - np.sqrt(np.log(2.0/delta)/(2.0*n))))
      print("Lambda is ", lbda)
  
  return lbda

def auditor_2(l,l_hat, alpha, delta, X_b, y_b,n,verbose=False):
  """ Auditor_2 ignores the square root term with delta.
    The goal of the auditor is to check for alpha consistency
    violations for a subgroup, that is it checks whether the 
    predicted mean of the subset deviates too much from the true mean.
    
    Inputs: 
    l - True label function
    l_hat - Predictor
    alpha - hyperparameter threshold for the auditor
    delta - hyperparameter
    X_b, y_b - Datapoints in a subgroup to be audited

    For mean_multicalibration l_hat is the prediction for each point and l=y_b

    Outputs: returns lbda
             lbda = +1 if predicted mean is overestimated
                    -1 if underestimated 
                    else None
  """
  lbda = None
  n_b = len(y_b)
  if n_b > 0:
    y_pred_avg = np.average(l_hat)
    y_true_avg = np.average(y_b) ##might be different with higher moments
    difference = y_pred_avg - y_true_avg
    adj_difference = abs(difference)
    denominator = (n_b / n)
    
    if denominator < alpha:
      # subgroup too small
      return lbda
    
    # consistency check
    if (adj_difference >= (alpha / denominator)):
      lbda = np.sign(difference)
    
    if verbose == True:
      print("Size of subgroup(n_b) is ", n_b, " Batch size(n) is ", n, " Proportion is ", (n_b / n))
      print('Difference between True and Predicted ', difference)
      print("Adjusted_difference(LHS term) is ", adj_difference)
      print("RHS term (alpha/denominator) is ", alpha / denominator)
      print("Lambda is ", lbda)
    return lbda