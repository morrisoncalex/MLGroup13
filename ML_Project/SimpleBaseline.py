import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold


def main():
	housing = pandas.read_csv('./housing.csv')
	housingDropped = pandas.read_csv('./housing.csv')
	
	#Deleting the categorical data
	housing.drop(columns=['ocean_proximity'], inplace=True)
	housingDropped.drop(columns=['ocean_proximity'], inplace=True)
	
	
	#creating housingDropped which is the data with rows with missing values removed
	missing = housingDropped.total_bedrooms.isna()
	missingI = []	
	for i in range(0,20639):
	    if missing[i]:
	        missingI.append(i)
	housingDropped.drop(missingI,axis=0,inplace=True)
	
	
	#now we have 2 sets of data - one with missing values and one with them removed
	#find the model which best regresses to predict the missing values
	#do this using the data with missing values removed
	match, fitModel = best_fit('total_bedrooms', housingDropped)
	
	#display the best field to regress with
	print("Best match  for total_bedrooms is: ")
	print(match)
	print('\n')
	
	#replace all missing values using the calculated model
	housing = replace_missing(housing, fitModel, match)
	
	
	
	#Get the model and data
	x, y, Model = create_model(housing)
	oldx, oldy, oldModel = create_model(housingDropped)
	
	#number of folds
	k = 5
	
	#training old and new models
	Model,       trainingAvg,    testingAvg = k_fold_train(Model, k, x, y)
	oldModel, oldTrainingAvg, oldTestingAvg = k_fold_train(Model, k, oldx, oldy)
	
	
	#PRINTING RESULTS
	
	#print its accuracy
	print('Average training error: ' + str(trainingAvg))
	print('Average testing error : ' + str(testingAvg))
	print('\n')
	
	#compare to data with missing values in it (old data)
	#print its accuracy
	print('Average training error (old data): ' + str(oldTrainingAvg))
	print('Average testing error  (old data): ' + str(oldTestingAvg))
	print('\n')
	
	#compute the difference in accuracy and print
	diffTraining = oldTrainingAvg - trainingAvg
	diffTesting  = oldTestingAvg  - testingAvg
	print('Training diff (old - new): ' + str(oldTrainingAvg - trainingAvg))
	print('Testing diff  (old - new): ' + str(oldTestingAvg - testingAvg))
	print('\n')
	
		
def create_model(housing):
	y    = housing.median_house_value.values.reshape(-1,1)
	x    = housing.drop(columns=['median_house_value'], inplace=False).values
	model = lm.LinearRegression()
	return x, y, model
		
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
	    
	    

	trainingAvg = trainingSum/k
	testingAvg = testingSum/k
	
	
	return Model, trainingAvg, testingAvg
	
	
	
def replace_missing(housing, Model, match):
	
	missing = housing.total_bedrooms.isna()
	
	for i in range(len(missing)):
		if missing[i]:
			housing['total_bedrooms'][i] = Model.predict(housing[match][i].reshape(-1,1))
			
		
	return housing
	
def best_fit(column, housing):
	#getting a list of the remaining fields
	types = list(housing)
	types.remove(column)
	types.remove('median_house_value')
	#print(types)
	
	#the values in the column we are finding a match for
	y = housing[column].values.reshape(-1,1)
	
	#setting up for fitting
	k = 3
	bestFit = 10000
	bestMatch = "none"
	Model = lm.LinearRegression()
	
	#TODO return the model to predict stuff
	bestModel = lm.LinearRegression()
	
	
	# Train a model against each column
	# compare model based on testing average and return the best one
	for i in range(len(types)):
		x = housing[types[i]].values.reshape(-1,1)
		Model, trainingAvg, testingAvg = k_fold_train(Model, k, x, y)
		if bestFit > testingAvg:
			bestFit = testingAvg
			bestMatch = types[i]
			bestModel = Model
		
		#PRINTINT MODEL RESULTS
		print("Fit between:")
		print(column)	
		print(types[i])
		print("1. Training Average:")
		print(trainingAvg)
		print("2. Testing Average:")
		print(testingAvg)
		print('\n')
		
	
	return bestMatch, bestModel










if __name__ == "__main__": main()







