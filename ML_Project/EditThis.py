
import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


# 1) Read the CSV.
# 2) Perform one hot encoding (or simply drop the 'ocean_proximity' column).
# 3) Delete rows with missing values.
# 4) Initialise variables for our Linear Regression model.
# 5) Min-Max Normalisation.
# 6) Perform k-fold training.
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
        housingColumns = list(housing.columns.values)
    else:
        # Drop the 'ocean_proximity' column, as it contains non-numeric data.
        housing.drop(columns=['ocean_proximity'], inplace=True)
        housingColumns = list(housing.columns.values)

    # 3) Replace missing data
    
    # To do this we need a copy of the data with rows with the missing values removed - we use this data to construct a model to predit the missing values
    housing_dropped = housing.copy(deep=True)
    #delete rows of missing values
    delete_missing(housing_dropped)
    #find the model to predict values
    match, fitModel = best_fit('total_bedrooms', housing_dropped)
    
    #TODO Delete prints
    #display the best field to regress with
    print("Best match  for total_bedrooms is: ")
    print(match)
    print('\n')
    
    #replace all missing values using the calculated model
    housing = replace_missing(housing, fitModel, match)
    
    #normalise the data
    normalised = np.array(housing)
    for i in range(0,9):
            v = normalised[:, i]   # foo[:, -1] for the last column
            normalised[:, i] = (v - v.min()) / (v.max() - v.min())
    
    print (normalised)
    housing = pandas.DataFrame(normalised)
    housing.columns = housingColumns
   # ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population','households', 'median_income','median_house_value', 'ocean_proximity']
        # print(df)
    
    
    # 4) Initialise Model, k, y, x and the degree of complexity.
    complexity = 2
    x, y, Model = create_model(housing)
    k = 5
    

    # 5) Perform k-fold training.
    Model, trainingAvgMSE, testingAvgMSE, trainingAvgR2, testingAvgR2 = k_fold_train(Model, k, x, y, complexity)

    print('Average MSE training error: ' + str(trainingAvgMSE))
    print('Average MSE testing error: ' + str(testingAvgMSE))
    print('Average R2 training error: ' + str(trainingAvgR2))
    print('Average R2 testing error: ' + str(testingAvgR2))

    determine_most_irrelevant_parameter(housing)

def delete_missing(data):
    missing = data.total_bedrooms.isna()
    missingI = []
    for i in range(0, 20639):
        if missing[i]:
            missingI.append(i)
    data.drop(missingI, axis=0, inplace=True)
    return data

def k_fold_train(Model, k, x, y, complexity):
    trainingSumMSE = 0
    testingSumMSE = 0
    trainingSumR2 = 0
    testingSumR2 = 0
    
    
    kf = KFold(n_splits=k, shuffle=True)
    
    for train_index, test_index in kf.split(x):
       x_train, x_test = x[train_index], x[test_index]
       y_train, y_test = y[train_index], y[test_index]
       Model.fit(x_train, y_train)
       
       y_train_pred = Model.predict(x_train)
       y_test_pred = Model.predict(x_test)
        
       trainingErrorMSE = mean_squared_error(y_train, y_train_pred)
       testingErrorMSE = mean_squared_error(y_test, y_test_pred)
       trainingSumMSE += trainingErrorMSE
       testingSumMSE += testingErrorMSE
       
       trainingErrorR2 = r2_score(y_train, y_train_pred)
       testingErrorR2 = r2_score(y_test, y_test_pred)
       trainingSumR2 += trainingErrorR2
       testingSumR2 += testingErrorR2
        
       print('MSE Training error: ' + str(trainingErrorMSE))
       print('MSE Testing error: ' + str(testingErrorMSE))
       print('r2 Training error: ' + str(trainingErrorR2))
       print('r2 Testing error: ' + str(testingErrorR2))

    trainingAvgMSE = trainingSumMSE/k
    testingAvgMSE = testingSumMSE/k
    trainingAvgR2 = trainingSumR2/k
    testingAvgR2 = testingSumR2/k
    
    steps_ridge = [('scalar', StandardScaler()), ('poly', PolynomialFeatures(degree=complexity)), ('model', Ridge(alpha=1, fit_intercept=True))]
    pipeline_ridge = Pipeline(steps_ridge)
    pipeline_ridge.fit(x_train, y_train)
    print('Training score Ridge regression : {}'.format(pipeline_ridge.score(x_train, y_train)))
    print('Test score ridge regression: {}'.format(pipeline_ridge.score(x_test, y_test)))
    
    return Model, trainingAvgMSE, testingAvgMSE, trainingAvgR2, testingAvgR2

#Function which initialises models and creates y and x
#Not used much, just makes Main() neater
def create_model(housing, predict='median_house_value'):
    y    = housing.median_house_value.values.reshape(-1, 1)
    x    = housing.drop(columns=[predict], inplace=False).values
    model = lm.LinearRegression()
    return x, y, model


# Function forms a model to predict missing values in a specified column (@param column)
# Model formed by regressing column against other columns and returning one with the best accuracy
# Function returns model and the name of matched column

def best_fit(column, housing):
    #getting a list of the remaining fields
    types = list(housing)
    types.remove(column)
    
    #the values in the column we are finding a match for
    y = housing[column].values.reshape(-1,1)
    
    #setting up for fitting
    k = 3
    bestFit = -10000
    bestMatch = "none"
    
    Model = lm.LinearRegression()
    bestModel = lm.LinearRegression()
    
    
    # Train a model against each column
    # compare model based on testing average and return the best one
    for i in range(len(types)):
        x = housing[types[i]].values.reshape(-1,1)
        Model, trainingAvg, testingAvg, unusedTestR2, unusedTrainR2 = k_fold_train(Model, k, x, y, 1)
        #1 = best possible fit = upper bound
        if bestFit < testingAvg:
            bestFit = testingAvg
            bestMatch = types[i]
            bestModel = Model
        
        #TODO Remove prints
        #PRINTINT MODEL RESULTS
        print("")
        print("Fit between:")
        print(column)    
        print(types[i])
        print("1. Training Average:")
        print(trainingAvg)
        print("2. Testing Average:")
        print(testingAvg)
        print('\n')
        
    
    return bestMatch, bestModel

# function which takes housing the data with the empty data cells plus the model and matching column in said model
# returns data witht the missing values replaced
def replace_missing(housing, Model, match):
    missing = housing.total_bedrooms.isna()
    
    for i in range(len(missing)):
        if missing[i]:
            housing['total_bedrooms'][i] = Model.predict(housing[match][i].reshape(-1,1))
            
        
    return housing

def determine_most_irrelevant_parameter(housing):
    columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    complexities = [1,2,3]
    
    # These are tables whose rows are the parameter we have dropped, and whose columns are the complexity/order of the regression, and the data in the corresponding cell is the accuracy of the model against either the training or testing data, respectively.
    training_accuracy = []
    testing_accuracy  = []

    k = 5   #Number of folds
    for col_index in range(0, len(columns)):
        housing_after_drop = housing.drop(columns=[columns[col_index], 'median_house_value'], inplace=False).values
        Model = lm.LinearRegression()

        train_acc_row = []
        test_acc_row = []

        for complexity in complexities:
            Model, _, _, next_train_acc, next_test_acc = k_fold_train(Model, k, housing_after_drop, housing['median_house_value'], complexity)
            train_acc_row.append(next_train_acc)
            test_acc_row.append(next_test_acc)

        training_accuracy.append(train_acc_row)
        testing_accuracy.append(test_acc_row)

    tablefmt='fancy_grid'

    print (tabulate(training_accuracy, tablefmt=tablefmt, headers=complexities, showindex=columns))
    print ('\n')
    print (tabulate(testing_accuracy, tablefmt=tablefmt, headers=complexities, showindex=columns))

if __name__ == "__main__": main()

