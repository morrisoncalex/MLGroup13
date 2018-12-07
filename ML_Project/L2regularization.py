import pandas

import numpy as np

import sklearn.linear_model as lm

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

housing = pandas.read_csv('./housing.csv')

# print(housing)


# Deleting the categorical data

housing.drop(columns=['ocean_proximity'], inplace=True)

# print(housing)

missing = housing.total_bedrooms.isna()

missingI = []

for i in range(0, 20639):

    if missing[i]:
        missingI.append(i)

housing.drop(missingI, axis=0, inplace=True)

y = housing.median_house_value.values.reshape(-1, 1)

x = housing.drop(columns=['median_house_value'], inplace=False).values

Model = lm.LinearRegression()

trainingSum = 0

testingSum = 0

k = 5

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

trainingAvg = trainingSum / k

testingAvg = testingSum / k

print('Average training error: ' + str(trainingAvg))

print('Average testing error: ' + str(testingAvg))


#sum of square of weights L2
# steps_ridge = [           #shrink size of weights
#
#     ('scalar', StandardScaler()),
#
#     ('poly', PolynomialFeatures(degree=3)),
#
#     ('model', Ridge(alpha=1, fit_intercept=True))
#
# ]
# pipeline_ridge = Pipeline(steps_ridge)
#
# pipeline_ridge.fit(x_train, y_train)
#
# print('Training score Ridge regression : {}'.format(pipeline_ridge.score(x_train, y_train)))
#
# print('Test score ridge regression: {}'.format(pipeline_ridge.score(x_test, y_test)))

# sum of weights L1
steps_lasso = [  # removes least important features

    ('scalar', StandardScaler()),

    ('poly', PolynomialFeatures(degree=3)),

    ('model', Lasso(alpha=1, fit_intercept=True))  # alpha is inversely proportional to variance
]

lasso_pipe = Pipeline(steps_lasso)

lasso_pipe.fit(x_train, y_train)

print('Training score Lasso: {}'.format(lasso_pipe.score(x_train, y_train)))

print('Test score Lasso: {}'.format(lasso_pipe.score(x_test, y_test)))
