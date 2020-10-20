

def auditor(l,l_hat, alpha, delta, X_b, y_b,n):
	# lbda is the lambda notation in the paper
  lbda = 0
  n_b = len(y_b)
  if n_b > 0:
    ## predicted values
    y_pred_avg = np.average(l_hat.predict(X_b))
    y_true_avg = np.average(y_b) ##might be different with higher moments
    difference = y_pred_avg - y_true_avg
    adj_difference = abs(difference)  - 2 * np.sqrt(np.log(2.0/delta)/(2.0*n_b)) #<-- make sure the second term is not too big, choosing alpha and delta
    # if (adj_difference >= (alpha/(n_b /n  - np.sqrt(np.log(2.0/delta)/(2.0*n_b))))):
    if (adj_difference >= alpha / ((n_b / n) - np.sqrt(np.log(2.0/delta)/(2.0*n)))):
      lbda = np.sign(difference)
  # print('n_b:', n_b)
  print(alpha)
  print('difference:', difference)
  # print('adj_difference:', adj_difference)
  return lbda