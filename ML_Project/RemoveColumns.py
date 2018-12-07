import pandas
#import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold


def main():
	housing = pandas.read_csv('./housing.csv')
	#print(housing)

	#Deleting the categorical data
	housing.drop(columns=['ocean_proximity'], inplace=True)
	#print(housing)
	missing = housing.total_bedrooms.isna()
	missingI = []	
	for i in range(0,20639):
	    if missing[i]:
	        missingI.append(i)
	housing.drop(missingI,axis=0,inplace=True)
	
	y = housing.median_house_value.values.reshape(-1,1)
	#x = housing.drop(columns=['median_house_value'], inplace=False).values
	c = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]

	for i in range(0,8):
		colDrop = housing.drop(columns=[c[i], 'median_house_value'], inplace=False).values
		Model = lm.LinearRegression()
		k = 5

		Model, trainingAvg, testingAvg = k_fold_train(Model, k, colDrop, y)
		print("dropped " + c[i] + ' Average training error: ' + str(trainingAvg))
		print("dropped " + c[i]+ ' Average testing error: ' + str(testingAvg))
	
		
def k_fold_train(Model, k, x, y):
	trainingSum = 0
	testingSum = 0
	
	
	kf = KFold(n_splits=k, shuffle=True)
	
	for train_index, test_index in kf.split(x):
	    x_train, x_test = x[train_index], x[test_index]
	    y_train, y_test = y[train_index], y[test_index]
	    Model.fit(x_train, y_train)
	    
	    trainingError = Model.score(x_train, y_train)
	    testingError = Model.score(x_test, y_test)
	    trainingSum += trainingError
	    testingSum += testingError
	    
	    print('Training error: ' + str(trainingError))
	    print('Testing error: ' + str(testingError))

	trainingAvg = trainingSum/k
	testingAvg = testingSum/k
	
	
	return Model, trainingAvg, testingAvg
	
	
	
	


if __name__ == "__main__": main()


