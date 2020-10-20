import parfit.parfit as pf
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

def learning_oracle_consistency_algorithm(x_test, y_pred, y_test, alpha, delta):
  pos_resid = y_pred - y_test['income_ <=50K'].tolist()
  neg_resid = -1 * pos_resid

 
  clf1 = SGDClassifier(loss = 'log') #bestModel(x_test, pos_resid, x_test, pos_resid)
  clf1.fit(x_test, pos_resid)

  y_test_resid_pos = clf1.predict(x_test)

  clf2 = SGDClassifier(loss = 'log') # bestModel(x_test, neg_resid, x_test, neg_resid)
  clf2.fit(x_test, neg_resid)

  y_test_resid_neg = clf2.predict(x_test)
  print(y_test_resid_neg)

  pos_high_error_set = []
  neg_high_error_set = []
 
  for idx in range(len(y_test_resid_pos)):
    if (y_test_resid_pos[idx] == 1):
      pos_high_error_set.append(x_test[idx])
    if (y_test_resid_neg[idx] == 1):
      neg_high_error_set.append(x_test[idx])
  return pos_high_error_set, neg_high_error_set