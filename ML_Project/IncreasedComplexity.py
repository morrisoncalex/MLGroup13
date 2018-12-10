import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

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
x = housing.drop(columns=['median_house_value'], inplace=False).values
#print(x.size)
complexityDegree = 2
poly = PolynomialFeatures(degree=complexityDegree)
x = poly.fit_transform(x)
#print(x.size)

Model = lm.LinearRegression()
trainingSumR2 = 0
testingSumR2 = 0
trainingSumMSE = 0
testingSumMSE = 0
trainingSumMAE = 0
testingSumMAE = 0
k = 5

kf = KFold(n_splits=k, shuffle=True)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Model.fit(x_train, y_train)
    y_train_pred = Model.predict(x_train)
    y_test_pred = Model.predict(x_test)
    
    trainingErrorR2 = r2_score(y_train,y_train_pred)
    testingErrorR2 = r2_score(y_test,y_test_pred)
    trainingSumR2 += trainingErrorR2
    testingSumR2 += testingErrorR2
    
    trainingErrorMSE = mean_squared_error(y_train,y_train_pred)
    testingErrorMSE = mean_squared_error(y_test,y_test_pred)
    trainingSumMSE += trainingErrorMSE
    testingSumMSE += testingErrorMSE
    
    trainingErrorMAE = mean_absolute_error(y_train,y_train_pred)
    testingErrorMAE = mean_absolute_error(y_test,y_test_pred)
    trainingSumMAE += trainingErrorMAE
    testingSumMAE += testingErrorMAE
    
    print('Training error: ' + str(trainingErrorR2))
    print('Testing error: ' + str(testingErrorR2))

trainingAvgR2 = trainingSumR2/k
testingAvgR2 = testingSumR2/k
print('Average R2 training error: ' + str(trainingAvgR2))
print('Average R2 testing error: ' + str(testingAvgR2))

trainingAvgMSE = trainingSumMSE/k
testingAvgMSE = testingSumMSE/k
print('Average MSE training error: ' + str(trainingAvgMSE))
print('Average MSE testing error: ' + str(testingAvgMSE))

trainingAvgMAE = trainingSumMAE/k
testingAvgMAE = testingSumMAE/k
print('Average MAE training error: ' + str(trainingAvgMAE))
print('Average MAE testing error: ' + str(testingAvgMAE))