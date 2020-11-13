# import pre_process_colnames as process
import re

def preprocess_colnames(categories, colnames):
  ## example: categories = ['race','gender']
    colnames = list(map(str,colnames))
    categories_dict = {}
    for name in categories:
      r = re.compile(name+'_')
      newlist = list(filter(r.match, colnames))
      categories_dict[name] = newlist
    return categories_dict

def projected_gradient_descent(T, eta, delta, l, X, y, y_pred, categories, num_buckets, subgroup_picker,verbose=False):
  update_list = []
  n = len(X)
  colnames = X.columns
  cat_dict = preprocess_colnames(categories, colnames)
  sum_lbda = 0
  for t in range(T):
    sub_picker, subgroups_str = subgroup_picker.subgroup_func_factory(cat_dict, num_buckets)
    indices = sub_picker(X,y_pred)
    # if subgroups_str == [[],[]]:
    #   print("TA DA")
    #   continue
    X_b, y_b, y_pred_b = X[indices], y[indices], y_pred[indices]
    lbda = auditor(l,y_pred_b, alpha, delta, X_b, y_b,n)
    print("lambda: ", lbda)
    if lbda == 0:
      sum_lbda = sum_lbda+1
    if lbda <= 0 or lbda > 0:
    # if lbda != 0:
      y_pred_old = y_pred.copy(deep=True)
      old = np.mean(y_pred_b)
      y_pred[indices] = y_pred[indices] - eta*lbda
      ## debug:
      true_sub = np.mean(y_b)
      modified_sub = np.mean(y_pred[indices])
      if verbose:
        print("old mean for subgroup: ", old, " True mean: ", true_sub, " Modified mean: ", modified_sub)
        print(str(lbda) + ':')
        print(subgroups_str)
        print('diff:' + str(sum(y_pred != y_pred_old)))
       
      update_list.append((sub_picker, eta*lbda))
      print("-------------------------------------next subgroup now")
  # project between 0 and 1 after all the iterations
  y_pred = y_pred.apply(lambda i : 0 if i <= 0 else min(i,1))
  print("num of instances when lbda = 0 is ", str(sum_lbda))
  # return the modified predictions
  return y_pred, update_list



