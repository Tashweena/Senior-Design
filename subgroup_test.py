from random import randrange,sample
import numpy as np
import itertools
import matplotlib.pyplot as plt

## returns all the subgroups
def iter_subgroups(categories_dict, num_buckets):
  '''returns all possible subgroup combinations'''
  all_subgroups = []
  for key in categories_dict:
    subgroup = []
    for val in categories_dict[key]:
      s = set()
      s.add(val)
      subgroup.append(s)
    subgroup.append(set())
    all_subgroups.append(subgroup)
  output_subgroups = cross_product(all_subgroups)
  for key in categories_dict:
    output_subgroups.append(set(categories_dict[key]))
  
  # print (list(itertools.product(output_subgroups, [1,2,3])))
  
  return (list(itertools.product(output_subgroups, [i for i in range(num_buckets)])))

def iter_subgroups_moment(categories_dict, num_buckets):
  all_subgroups = []
  for key in categories_dict:
    subgroup = []
    for val in categories_dict[key]:
      s = set()
      s.add(val)
      subgroup.append(s)
    subgroup.append(set())
    all_subgroups.append(subgroup)
  output_subgroups = cross_product(all_subgroups)
  for key in categories_dict:
    output_subgroups.append(set(categories_dict[key]))
  
  mu_var_combinations = (list(itertools.product([i for i in range(num_buckets)], [i for i in range(num_buckets)])))
  return (list(itertools.product(output_subgroups, mu_var_combinations)))

def iter_subgroups_linear_data(num_subgroups, num_buckets):
  size = (1.0/num_subgroups) 
  all_subgroups = []
  incrementer = 0
  for i in range(num_subgroups):
    all_subgroups.append((incrementer, incrementer+size))
    incrementer = incrementer+size
  return (list(itertools.product(all_subgroups, [i for i in range(num_buckets)])))

def iter_subgroups_linear_data_moment(num_subgroups, num_buckets):
  size = (1.0/num_subgroups) 
  all_subgroups = []
  incrementer = 0
  for i in range(num_subgroups):
    all_subgroups.append((incrementer, incrementer+size))
    incrementer = incrementer+size

  mu_var_combinations = (list(itertools.product([i for i in range(num_buckets)], [i for i in range(num_buckets)])))
  return (list(itertools.product(all_subgroups, mu_var_combinations)))


## does the same thing as before
def all_subgroup_func_factory_crime(subgroups, num_buckets, bucket):
    picked_subgroups = []
    prefix = subgroups[0][:3]
    first = []
    for s in subgroups:
      if s[:3] == prefix:
        first.append(s)
      else:
        picked_subgroups.append([s])
    picked_subgroups.append(first)
    # bucket range
    size = (1.0/num_buckets)
    lower = size * bucket
    upper = lower + size

    def subgroup_picker(x_test,y_pred):
      indices = np.array([True]*len(x_test))
      for lst in picked_subgroups:
        temp = np.array([False]*len(x_test))
        if len(lst) == 0:
          # print('past')
          continue
        else:
          for ele in lst:
            index = np.array(x_test.apply(lambda x : True if x[ele]==1 else False, axis=1).values)
            temp = temp | index
          indices = indices & temp
      ## decide indices for the bucket
      ## make sure y_pred is a pd.Series
      index = np.array(y_pred.apply(lambda y : True if (y > lower and y<=upper) else False))
      # print(sum(index))
      indices = indices & index
      return indices

    def everything_picker(x_test,y_pred):
      indices = np.array([True]*len(x_test))
      return indices

    # return subgroup_picker, picked_subgroups
    return subgroup_picker, picked_subgroups 

def cross_product(lst):
  if len(lst) == 1:
    return lst[0]
  elif len(lst) == 2:
    # print(lst[0])
    # print(lst[1])

    cross = list(itertools.product(lst[0],lst[1]))
    # print(cross)
    out = []
    for ele in cross:
      first = ele[0]
      second = ele[1]
      if len(first) == 0:
        out.append(second)
      else:
        merge = first.union(second)
        out.append(merge)
    return out
  else:
    n = len(lst)
    middle = int(n/2)
    left = lst[:middle]
    right = lst[middle:]
    cross_left = cross_product(left)
    cross_right = cross_product(right)
    cross = itertools.product(cross_left, cross_right)
    out = []
    for ele in cross:
      first = ele[0]
      second = ele[1]
      if len(first) == 0:
        out.append(second)
      else:
        merge = first.union(second)
        out.append(merge)
    return out


def all_subgroup_func_factory(subgroups, num_buckets, bucket, data=None, beta = None):
  if data == 'crime':
    subgroup_picker, picked_subgroups = all_subgroup_func_factory_crime(subgroups, num_buckets,bucket)

    return subgroup_picker, picked_subgroups
  elif data == 'cifar':
    subgroup_picker, picked_subgroups = all_subgroup_func_factory_cifar(subgroups, num_buckets, bucket)
    return subgroup_picker, picked_subgroups
  elif data == 'health':
    subgroup_picker = all_subgroup_func_factory_health(subgroups, num_buckets, bucket)
    return subgroup_picker
  else:
    subgroup_picker = all_subgroup_func_factory_oracle(num_buckets, bucket, beta)
    return subgroup_picker


def subgroup_func_factory_crime(categories_dict, num_buckets):
    picked_subgroups = []
    for key in categories_dict:
      subgroups = categories_dict[key]
      choice = randrange(len(subgroups))
      chosen_groups = sample(subgroups, choice)
      picked_subgroups.append(chosen_groups)
    # print(picked_subgroups)
    ## then, we pick a bucket
    # empty = True
    # for p in picked_subgroups:
    #   empty = empty & (len(p) == 0) 
    # if  empty:
    #   print("EMPTY SUBGROUPS ", picked_subgroups)
    bucket_num = randrange(num_buckets)
    print("the chosen bucket is ", bucket_num)
    size = (1.0/num_buckets)
    lower = size * bucket_num
    upper = lower + size
    # print(bucket_num, lower, upper)
    def subgroup_picker(x_test,y_pred):
      indices = np.array([True]*len(x_test))
      for lst in picked_subgroups:
        temp = np.array([False]*len(x_test))
        if len(lst) == 0:
          # print('past')
          continue
        else:
          for ele in lst:
            index = np.array(x_test.apply(lambda x : True if x[ele]==1 else False, axis=1).values)
            temp = temp | index
          indices = indices & temp
      ## decide indices for the bucket
      ## make sure y_pred is a pd.Series
      index = np.array(y_pred.apply(lambda y : True if (y > lower and y<=upper) else False))
      # print(sum(index))
      indices = indices & index
      return indices

    return subgroup_picker, picked_subgroups


def label_subgroup_converter(subgroups):
  labels = set()
  mapping = {'in_water': {6,8},'out_water':{0,1,2,3,4,5,7,9}, 'is_animal':{2,3,4,5,6,7}, 'is_not_animal':{0,1,8,9}}
  picked_subgroups = []
  prefix = subgroups[0][-3:]
  first = []
  for s in subgroups:
    label = list(mapping[s])
    if s[-3:] == prefix:
      first.extend(label)
    else:
      picked_subgroups.append(label)
  picked_subgroups.append(first)

  # for subgroup in subgroups:
  #   labels.union(mapping[subgroup])
  return picked_subgroups

def all_subgroup_func_factory_health(subgroups, num_buckets, bucket):
  invert_dict = {'black':('race', 1), 'white':('race', 0), 'female': ('dem_female',1), 'male':('dem_female',0)}
  race_lst = []
  gender_lst = []
  size = (1.0/num_buckets)
  lower = size * bucket
  upper = lower + size
  for subgroup in subgroups:
    cat = invert_dict[subgroup][0]
    if cat == 'race':
      race_lst.append(subgroup)
    else:
      gender_lst.append(subgroup)
  
  def subgroup_picker(z, y_pred):
    indices = np.array([True]*len(y_pred))
    if len(race_lst) == 0:
      race_indices = np.array([True]*len(y_pred))
    else: 
      race_indices = np.array([False]*len(y_pred))
      for ele in race_lst:
        num = invert_dict[ele][1]
        sub_indices = z['race'] == num
        sub_indices = sub_indices.values
        race_indices = race_indices | sub_indices
    if len(gender_lst) == 0:
      gender_indices = np.array([True]*len(y_pred))
    else: 
      gender_indices = np.array([False]*len(y_pred))
      for ele in gender_lst:
        num = invert_dict[ele][1]
        sub_indices = z['dem_female'] == num
        sub_indices = sub_indices.values
        gender_indices = gender_indices | sub_indices
    buck_indices = np.array(y_pred.apply(lambda y : True if (y >= lower and y<upper) else False))
      # print(sum(index))
    # print("Gender indices in subgroup picker,", sum(gender_indices))
    indices = indices & buck_indices & race_indices & gender_indices
    # indices = indices.values
    return indices
  return subgroup_picker
  
def all_subgroup_func_factory_cifar(subgroups, num_buckets, bucket):
  '''
  Return a subgroup_picker function that return a list of boolean to indicate whether
  each row in the df belong to a specified subgroup.

  subgroups: the chosen subgroup
  num_buckets: divide the y_pred value (between [0,1]) into num_buckets of intervals
  bucket: only look at y_pred in that interval
  '''
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  labels = label_subgroup_converter(subgroups)
  size = (1.0/num_buckets)
  lower = size * bucket
  upper = lower + size
  
  def subgroup_picker(reference, y_pred):
    '''
    Return a boolean list same length as y_pred. True if the data (i.e. X_i) corresponding to y_pred_i 
    belongs to the specified subgroup and bucket interval

    reference: the X values that include indicator columns of each X_i's subgroup membership
    '''
    indices = np.array([True]*len(reference))
    for label in labels:
      sub_indices = np.array([False]*len(reference))
      for ele in label:
        index = reference == ele
        sub_indices = sub_indices | index
      indices = indices & sub_indices
    ## decide indices for the bucket
    ## make sure y_pred is a pd.Series
    buck_index = np.array(y_pred.apply(lambda y : True if (y > lower and y<=upper) else False))
    # print(sum(index))
    indices = indices & buck_index
    return indices
  def everything_picker(x_test,y_pred):
      indices = np.array([True]*len(x_test))
      return indices
  return subgroup_picker, subgroups
  # return everything_picker, subgroups

def all_subgroup_func_factory_cifar_var(subgroups, num_buckets, bucket_mu, bucket_var):
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  labels = label_subgroup_converter(subgroups)
  size = (1.0/num_buckets)
  lower_mu = size * bucket_mu
  upper_mu = lower_mu + size

  lower_var = size * bucket_var
  upper_var = lower_var + size 
  def subgroup_picker(reference, calibrated_means, calibrated_var):
      indices = np.array([True]*len(reference))
      for label in labels:
        sub_indices = np.array([False]*len(reference))
        for ele in label:
          index = reference == ele
          sub_indices = sub_indices | index
        indices = indices & sub_indices
      ## decide indices for the bucket
      ## make sure y_pred is a pd.Series
      buck_index_mu = np.array(calibrated_means.apply(lambda y : True if (y >= lower_mu and y<upper_mu) else False))
      buck_index_var = np.array(calibrated_var.apply(lambda y : True if (y >= lower_var and y<upper_var) else False))
      # print("LM",lower_mu)
      # print("UM",upper_mu)
      # print("LV",lower_var)
      # print("UV",upper_var)
      # print("calibrated means", calibrated_means)
      # print("calibrated var", calibrated_var)

      # print("buck_index_mu", sum(buck_index_mu))
      # print("buck_index_var", sum(buck_index_var))
      # print("batch size", len(reference))
      # print("indices", str(indices))

      indices = indices & buck_index_mu & buck_index_var
      # print("sum indices", sum(indices))
      return indices
  def everything_picker(reference, calibrated_means, calibrated_var):
      indices = np.array([True]*len(reference))
      return indices
  return subgroup_picker, subgroups
  # return everything_picker, subgroups

def all_subgroup_linear_data_mean(subgroups, num_buckets, bucket):
  labels = subgroups
  size = (1.0/num_buckets)
  lower_mu = size * bucket
  upper_mu = lower_mu + size

  def subgroup_picker(reference, y_pred):
      indices = np.ones(reference.shape, dtype=bool)
      sub_indices = np.zeros(reference.shape, dtype=bool)
    
      (lower_bound, upper_bound) = labels
      index = np.logical_and(reference < upper_bound, reference >= lower_bound) 
      sub_indices = sub_indices | index
      indices = indices & sub_indices
      ## decide indices for the bucket
      ## make sure y_pred is a pd.Series
      buck_index = np.array(y_pred.apply(lambda y : True if (y >= lower_bound and y<upper_bound) else False))
      buck_index = buck_index.reshape((buck_index.shape[0], 1))
      indices = indices & buck_index
      return indices
  def everything_picker(x_test,y_pred):
      indices = np.array([True]*len(x_test))
      return indices
  return subgroup_picker, subgroups

def all_subgroup_linear_data_var(subgroups, num_buckets, bucket_mu, bucket_var):

  labels = subgroups
  size = (1.0/num_buckets)
  lower_mu = size * bucket_mu
  upper_mu = lower_mu + size

  lower_var = size * bucket_var
  upper_var = lower_var + size 
  
  def subgroup_picker(reference, calibrated_means, calibrated_var):
      indices = np.ones(reference.shape, dtype=bool)
      sub_indices = np.zeros(reference.shape, dtype=bool)
      (lower_bound, upper_bound) = labels
      index = np.logical_and(reference < upper_bound, reference >= lower_bound) 
      sub_indices = sub_indices | index
      # import pdb
      # pdb.set_trace()
      indices = indices & sub_indices
      ## decide indices for the bucket
      ## make sure y_pred is a pd.Series
      buck_index_mu = np.array(calibrated_means.apply(lambda y : True if (y > lower_mu and y<=upper_mu) else False))
      buck_index_var = np.array(calibrated_var.apply(lambda y : True if (y >= lower_var and y<upper_var) else False))
      buck_index_mu = buck_index_mu.reshape((buck_index_mu.shape[0], 1))
      buck_index_var = buck_index_var.reshape((buck_index_var.shape[0], 1))

      indices = indices & buck_index_mu & buck_index_var
      return indices
  def everything_picker(reference, calibrated_means, calibrated_var):
      indices = np.array([True]*len(reference))
      return indices
  return subgroup_picker, subgroups
  #return everything_picker, subgroups


### subgroup picker for beta - learning oracle
#define the sigmoid function
def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))


def bucket_divider(num_buckets, bucket,y_pred):
  size = (1.0/num_buckets)
  lower = size * bucket
  upper = lower + size
  buck_index = np.array(y_pred.apply(lambda y : True if (y >= lower and y<upper) else False))
  return buck_index


def all_subgroup_func_factory_oracle(num_buckets, bucket, beta):
  size = (1.0/num_buckets)
  lower = size * bucket
  upper = lower + size
  def sub_picker_learning_oracle(x_val, y_pred):
    # print('x_val: ', x_val.shape)
    # print(x_val)
    buck_index = np.array(y_pred.apply(lambda y : True if (y >= lower and y<upper) else False))
    # print('buck_index len:', len(buck_index))
    buck_x_val = x_val[buck_index,:]
    # print(buck_x_val.shape)
    # print(len(buck_index))
    buck_x_val = np.concatenate((buck_x_val, np.ones((len(buck_x_val), 1))), 1)
    # buck_residual = sigmoid(np.matmul(buck_x_val,beta)) ##### FOR LOGISTIC LOSS
    buck_residual = np.matmul(buck_x_val,beta)
    # print("Residuals predicted: ", buck_residual)
    # plt.plot(buck_residual)
    indices = []
    counter = 0
    for idx in range(len(x_val)):
      if buck_index[idx] ==1:
        ## if we have only 2 beta
        # if(buck_residual[counter] > .1):
        if((buck_residual[counter] > .1) or (buck_residual[counter] < -0.1)):
          indices.append(True)
        else:
          indices.append(False)
        counter += 1
      else:
        indices.append(False)
    indices = np.array(indices)
    # print('indices len:', len(indices))
    return indices
  return sub_picker_learning_oracle





# from random import randrange,sample
# import numpy as np

# def subgroup_func_factory(categories_dict, num_buckets):
#     picked_subgroups = []
#     for key in categories_dict:
#       subgroups = categories_dict[key]
#       choice = randrange(len(subgroups))
#       chosen_groups = sample(subgroups, choice)
#       picked_subgroups.append(chosen_groups)
#     # print(picked_subgroups)
#     ## then, we pick a bucket
#     # empty = True
#     # for p in picked_subgroups:
#     #   empty = empty & (len(p) == 0) 
#     # if  empty:
#     #   print("EMPTY SUBGROUPS ", picked_subgroups)
#     bucket_num = randrange(num_buckets)
#     print("the chosen bucket is ", bucket_num)
#     size = (1.0/num_buckets)
#     lower = size * bucket_num
#     upper = lower + size
#     # print(bucket_num, lower, upper)
#     def subgroup_picker(x_test,y_pred):
#       indices = np.array([True]*len(x_test))
#       for lst in picked_subgroups:
#         temp = np.array([False]*len(x_test))
#         if len(lst) == 0:
#           # print('past')
#           continue
#         else:
#           for ele in lst:
#             index = np.array(x_test.apply(lambda x : True if x[ele]==1 else False, axis=1).values)
#             temp = temp | index
#           indices = indices & temp
#       ## decide indices for the bucket
#       ## make sure y_pred is a pd.Series
#       index = np.array(y_pred.apply(lambda y : True if (y > lower and y<=upper) else False))
#       # print(sum(index))
#       indices = indices & index
#       return indices
#     return subgroup_picker, picked_subgroups