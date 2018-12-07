

import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def main():
    housing = pandas.read_csv('./housing.csv')
    missing = housing.total_bedrooms.isna()
    missingI = []    
    for i in range(0,20639):
        if missing[i]:
            missingI.append(i)
    housing.drop(missingI,axis=0,inplace=True)
    normalised = np.array(housing)
   
    for i in range(0,9):
        v = normalised[:, i]   # foo[:, -1] for the last column
      #  print(v)
        normalised[:, i] = (v - v.min()) / (v.max() - v.min())
        
   # print (normalised)
    df = pd.DataFrame(normalised)
    df.columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population','households', 'median_income','median_house_value', 'ocean_proximity']
   # print(df)
     
    #Deleting the categorical data
    df.drop(columns=['ocean_proximity'], inplace=True)
  #  df.drop(columns=['median_house_value'], inplace=True)
    print(df)
    #print(housing)
   
    y = df.median_house_value.values.reshape(-1,1)
    x = df.drop(columns=['median_house_value'], axis = 1 , inplace=False).values
    
    Model = lm.LinearRegression()
    k = 5
    
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
    
    
    
    


if __name__ == "__main__": main()
