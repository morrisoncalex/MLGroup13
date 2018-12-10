
import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


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
    
    
    # 4) Initialise Model, k, y and x.
    x, y, Model = create_model(housing)
    k = 5
    

    # 5) Perform k-fold training.
    Model, trainingAvg, testingAvg = k_fold_train(Model, k, x, y)

    print('Average training error: ' + str(trainingAvg))
    print('Average testing error: ' + str(testingAvg))

def delete_missing(data):
    missing = data.total_bedrooms.isna()
    missingI = []
    for i in range(0, 20639):
        if missing[i]:
            missingI.append(i)
    data.drop(missingI, axis=0, inplace=True)
    return data

def k_fold_train(Model, k, x, y):
    trainingSum = 0
    testingSum = 0
    
    
    kf = KFold(n_splits=k, shuffle=True)
    
    for train_index, test_index in kf.split(x):
       x_train, x_test = x[train_index], x[test_index]
       y_train, y_test = y[train_index], y[test_index]
       Model.fit(x_train, y_train)
       y_train_pred = Model.predict(x_train)
       y_test_pred = Model.predict(x_test)
        
       trainingError = mean_squared_error(y_train, y_train_pred)
       testingError = mean_squared_error(y_test, y_test_pred)
       trainingSum += trainingError
       testingSum += testingError
        
       print('Training error: ' + str(trainingError))
       print('Testing error: ' + str(testingError))

    trainingAvg = trainingSum/k
    testingAvg = testingSum/k
    
    
    return Model, trainingAvg, testingAvg
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
        Model, trainingAvg, testingAvg = k_fold_train(Model, k, x, y)
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

# function whih takes housing the data with the empty data cells plus the model and matching column in said model
# returns data witht the missing values replaced
def replace_missing(housing, Model, match):
    missing = housing.total_bedrooms.isna()
    
    for i in range(len(missing)):
        if missing[i]:
            housing['total_bedrooms'][i] = Model.predict(housing[match][i].reshape(-1,1))
            
        
    return housing
    

if __name__ == "__main__": main()

