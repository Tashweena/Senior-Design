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