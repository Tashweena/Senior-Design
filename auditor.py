import numpy as np
def auditor(l,l_hat, alpha, delta, X_b, y_b,n):
  # optional parameter take 
  # l_hat = y_pred
  # lbda is the lambda notation in the paper
  lbda = None
  n_b = len(y_b)
  if n_b > 0:
    ## predicted values
    y_pred_avg = np.average(l_hat)
    y_true_avg = np.average(y_b) ##might be different with higher moments
    difference = y_pred_avg - y_true_avg
    adj_difference = abs(difference)  - 2 * np.sqrt(np.log(2.0/delta)/(2.0*n_b))    
    print('difference:', abs(difference), " adjusted_difference ", adj_difference)

    # if (abs(difference) <= tol):
    #   return lbda
    denominator = ((n_b / n) - np.sqrt(np.log(2.0/delta)/(2.0*n)))
    if denominator < alpha:
      return lbda
    if (adj_difference >= alpha / denominator):
      lbda = np.sign(difference)
    # print('difference:', abs(difference))
    # print("n' is ", n_b)
    # print(" n is ", n)
    # print(" alpha/denominator ", alpha / ((n_b / n) - np.sqrt(np.log(2.0/delta)/(2.0*n))))
    # print(" square root term ", 2 * np.sqrt(np.log(2.0/delta)/(2.0*n_b)))
    # print("RHS ", (alpha / ((n_b / n) - np.sqrt(np.log(2.0/delta)/(2.0*n)))) + 2 * np.sqrt(np.log(2.0/delta)/(2.0*n_b)) )
    # print("n'/n_b ", (n_b / n))
    # print(" second term ",np.sqrt(np.log(2.0/delta)/(2.0*n)))
    
    # print("LHS second term ", - 2 * np.sqrt(np.log(2.0/delta)/(2.0*n_b)))
    # print('adj_difference:', adj_difference)
  # print('n_b:', n_b)
  # print(alpha)
  return lbda


## this auditor ignores the square root term
def auditor_2(l,l_hat, alpha, delta, X_b, y_b,n):
  # print("Calling this -------------------------------------------------")
  lbda = None
  n_b = len(y_b)
  if n_b > 0:
    ## predicted values
    y_pred_avg = np.average(l_hat)
    y_true_avg = np.average(y_b) ##might be different with higher moments
    difference = y_pred_avg - y_true_avg
    adj_difference = abs(difference)
    # if (abs(difference) <= tol):
    #   return lbda
    denominator = (n_b / n)
    print('difference:', abs(difference), " adjusted_difference ", adj_difference)
    # print("denominator ", denominator)
    # print("alpha/denominator ", alpha / denominator)
    if denominator < alpha:
      # print("Denominator is small")
      return lbda
    # print("Result is.  -------", adj_difference >= (alpha / denominator))
    if (adj_difference >= (alpha / denominator)):
      lbda = np.sign(difference)
    return lbda



# def auditor_2(l,l_hat, alpha, delta, X_b, y_b,n):
#   lbda = None
#   n_b = len(y_b)
#   if n_b > 0:
#     ## predicted values
#     y_pred_avg = np.average(l_hat)
#     y_true_avg = np.average(y_b) 
#     difference = y_pred_avg - y_true_avg
#     adj_difference = abs(difference)
#     denominator = (n_b / n)
#     if denominator < alpha:
#       return lbda
#     if (adj_difference >= (alpha / denominator)):
#       lbda = np.sign(difference)
#     return lbda





