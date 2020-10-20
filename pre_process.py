import numpy as np
import pandas as pd
import scipy 
import io
import requests
from matplotlib import pyplot


def pre_process():
	url="http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
	s=requests.get(url).content
	data = pd.read_csv(url, header=None)

	data.rename(columns={0: 'age', 1: 'workclass', 2: 'fnlwgt',3: 'education',4: 'educational-num',5: 'marital',6: 'occupation', 7: 'relationship', 8: 'race', 9: 'gender', 10: 'capital gain', 11: 'capital loss',12: 'hours per week', 13: 'country', 14: 'income' }, inplace=True)
	data['country'] = data['country'].replace(' ?',np.nan)
	data['workclass'] = data['workclass'].replace(' ?',np.nan)
	data['occupation'] = data['occupation'].replace(' ?',np.nan)
	#dropping the NaN rows now 
	data.dropna(how='any',inplace=True)

	modified_df = one_hot_encoding(data)

	# split into x and y
	df_y = pd.DataFrame(modified_df['income_ <=50K'])
	df_x = modified_df.copy()
	df_x.drop({'income_ <=50K', 'income_ >50K'}, inplace=True, axis=1)

	return df_x,df_y


def one_hot_encoding(data):
	cols = data.columns
	df = pd.DataFrame()
	for col in cols:
	   if data[col].dtype == 'int64':
	    df[col] = data[col]
	   else:
	    temp = pd.get_dummies(data[col], prefix=col)
	    df = pd.concat([temp,df], axis=1)
	return df