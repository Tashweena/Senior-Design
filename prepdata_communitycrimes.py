# -*- coding: utf-8 -*-
"""PrepData_CommunityCrimes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19WbgqmtL8Fpgqug1rNk4mksU98h9uFkQ
"""

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor

def prep_data():
  ### DIFFERENT DATASET
  url="https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
  s=requests.get(url).content
  data = pd.read_csv(url, header=None)
  # data.head()
  clean_data = data.iloc[:, 5:128].drop(columns=range(27,33)).replace('?',np.nan).dropna(how='any', axis=1)
  df_x = clean_data.drop(columns=[127])
  df_y = clean_data.iloc[:,-1]
  x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.6, random_state=42)
  # regr = RandomForestRegressor(max_depth=10, max_features='sqrt', random_state=0, max_samples=0.5)
  # regr.fit(x_train, y_train)
  # y_pred = regr.predict(x_test)
  # y_pred = pd.Series(y_pred)
  x_test['race_nonwhite'] = x_test[8].apply(lambda x : 1 if x < 0.4 else 0)
  x_test['race_white'] = x_test[8].apply(lambda x : 1 if x >= 0.4 else 0)
  x_test['edu_highschool'] = x_test[35].apply(lambda x : 1 if x < 0.7 else 0)
  x_test['edu_nonehighschool'] = x_test[35].apply(lambda x : 1 if x >= 0.7 else 0)
  return x_test, y_test,x_train,y_train


