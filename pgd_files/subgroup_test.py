from random import randrange,sample
import numpy as np
import itertools

## returns all the subgroups
def iter_subgroups(categories_dict, num_buckets):
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

## does the same thing as before
def all_subgroup_func_factory(subgroups, num_buckets,bucket):
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


def subgroup_func_factory(categories_dict, num_buckets):
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