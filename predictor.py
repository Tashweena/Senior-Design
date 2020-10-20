
# Predictors

from sklearn.model_selection import train_test_split 
from sklearn import tree

def predictor_decision_tree(df_x,df_y,test_size):
	x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size, random_state=42)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(x_train, y_train)
return clf
