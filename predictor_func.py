import pandas as pd
import numpy as np
class Predictor:
	def __init__(self, initial_predictor,x_train,y_train):

		self.initial_predictor = initial_predictor
		self.initial_predictor.fit(x_train,y_train)
		self.lbda_group_list = []		

	def predict_y(self, x_test, x_test_sub):
		y_pred = self.initial_predictor.predict(x_test)
		y_pred = pd.Series(y_pred)
		for func, val in self.lbda_group_list:
			indices = func(x_test_sub, y_pred)
			y_pred[indices] += val
			y_pred[indices] = y_pred[indices].apply(lambda i : 0 if i <= 0 else min(i,1))
			# print("The NEW mean is ")
			# print(np.mean(y_pred[indices]))
			## DO THE PROJECTION HERE

		return y_pred

	# amount is eta*lbda
	def update(self, sub_picker, amount):
		print("UPDATE IS BEING CALLED")
		self.lbda_group_list.append((sub_picker,amount))
		# print(self.lbda_group_list)
