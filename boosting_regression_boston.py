from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

# Load the boston data
boston_dat = load_boston()
# The data are in a dictonary -> print the keys
print(boston_dat.keys())
# Print the data description
print(boston_dat.DESCR)
# Inspect the shape of the data
print(boston_dat.data.shape)

### Now get the data and store them as Pandas data frame
boston = pd.DataFrame(boston_dat.data, columns=boston_dat.feature_names)
print(boston.head())

# The target (y) is "- MEDV     Median value of owner-occupied homes in $1000's", we add this to our data frame
boston['MEDV'] = boston_dat.target
print(boston.head())
print(boston.shape)

# Now define X, y
y = boston['MEDV']
X = boston.drop('MEDV', 1)
# Inspect shape of data
print(X.shape)
print(y.shape)

# Make a test and train set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=7)

# Standardise data
mean = xtrain.mean(axis=0)
xtrain -= mean
std = xtrain.std(axis=0)
xtrain /= std
xtest -= mean
xtest /= std

# SET UP LIGHTGBM
import lightgbm as lgb
import random
from sklearn.metrics import mean_squared_error

# Create dataset for lightgbm
lgb_train = lgb.Dataset(xtrain, ytrain)
lgb_eval = lgb.Dataset(xtest, ytest, reference=lgb_train)

# Perform random search for best hyperparameter
best_mse = 99999999999999999
best_params = []
# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
for i in range(0,100):
    print("Training progress: %i" %(i), end="\r")
    params = {
            'objective': 'regression',
            'learning_rate' : random.uniform(0.6, 0.9), 
            'metric': 'mean_squared_error',
            'seed': 7,
            'verbose': -1,
            'boosting_type' : 'gbdt'
        }

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100000,
                    valid_sets=lgb_eval,
                    verbose_eval=False,
                    early_stopping_rounds=100)

    # predict (on test)
    ypredlgbm = gbm.predict(xtest, num_iteration=gbm.best_iteration)
    lgb_mse = mean_squared_error(ytest, ypredlgbm)
    if lgb_mse<best_mse:
        best_mse = lgb_mse
        best_params = params


print("========================================")
print("MSE LightGBM: %s" %best_mse)
print(best_params)
print("========================================")

# OLS regression as baseline model
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(xtrain, ytrain)
yhat = reg.predict(np.array(xtest))
print("MSE OLS:      %s" %mean_squared_error(ytest, yhat))
