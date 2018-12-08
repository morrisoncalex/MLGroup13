import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold


# 1) Read the CSV.
# 2) Perform one hot encoding (or simply drop the 'ocean_proximity' column).
# 3) Delete rows with missing values.
# 4) Initialise variables for our Linear Regression model.
# 5) Perform k-fold training.
def main():
	# TOGGLES
	is_one_hot_encoding_enabled = True

	# 1) Read the CSV. The variable 'housing' is of type 'DataFrame'.
	housing = pandas.read_csv('./housing.csv')

	# 2) Decide if to perform one hot encoding.
	if is_one_hot_encoding_enabled:
		# Convert 'ocean_proximity' into multiple columns with names like ocean_proximity_INLAND, ocean_proximity_ISLAND,
		# each of which contains numeric data.
		housing = pandas.get_dummies(housing, prefix=['ocean_proximity'])
	else:
		# Drop the 'ocean_proximity' column, as it contains non-numeric data.
		housing.drop(columns=['ocean_proximity'], inplace=True)

	# TODO: Tom Linker - complete missing data. For now, we're simply removing it.
	# 3) Delete rows with missing values.
	missing = housing.total_bedrooms.isna()
	missingI = []
	for i in range(0, 20639):
		if missing[i]:
			missingI.append(i)
	housing.drop(missingI, axis=0, inplace=True)

	# 4) Initialise Model, k, y and x.
	Model = lm.LinearRegression()
	k = 5
	y = housing.median_house_value.values.reshape(-1, 1)
	# We've used the 'median_house_value' column for y, so we drop it from x.
	x = housing.drop(columns=['median_house_value'], inplace=False).values

	# 5) Perform k-fold training.
	Model, trainingAvg, testingAvg = k_fold_train(Model, k, x, y)

	print('Average training error: ' + str(trainingAvg))
	print('Average testing error: ' + str(testingAvg))


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

	trainingAvg = trainingSum / k
	testingAvg = testingSum / k

	return Model, trainingAvg, testingAvg


if __name__ == "__main__": main()

