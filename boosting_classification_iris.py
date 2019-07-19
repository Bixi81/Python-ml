from sklearn import datasets

# Import data (from sklearn module)
iris = datasets.load_iris()
# Set x and target (y)
x = iris.data
y = iris.target

#########################################
# Split test/train
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=7)

#########################################
# Logit 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(xtrain, ytrain)
yhat = lr.predict(xtest)

# Results as data frame 
import pandas as pd
logres = pd.DataFrame({'test':ytest, 'pred':yhat})
# Recode correct prediction as 1 and incorrect as 0
logres['correct'] = 0
logres.loc[logres.test == logres.pred, 'correct'] = 1
print("Logit results:")
print(logres['correct'].value_counts())
print("================================")

#########################################################
# LightGBM
# https://lightgbm.readthedocs.io/en/latest/
import lightgbm as lgb
import random

# Create dataset for lightgbm
lgb_train = lgb.Dataset(xtrain, ytrain)
lgb_eval = lgb.Dataset(xtest, ytest, reference=lgb_train)

# Perform random search for best hyperparameter
best_params = []
best_correct_classes = 0
# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
for i in range(0,100):
    params = {
            # Parameter expl. https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst
            'objective' :'multiclass',# type of problem
            'num_class' : 3, # number of classes
            'learning_rate' : random.uniform(0.01, 10), # default = 0.1, type = double, aliases: shrinkage_rate, eta, constraints: learning_rate > 0.0
            'lambda_l1': random.uniform(0, 10), # L1 regularization (>=0)
            'metric': 'multi_logloss', # loss function
            'feature_fraction': random.uniform(0.01, 0.99),
            'bagging_fraction': random.uniform(0.01, 0.99),
            'bagging_freq': random.randint(0,100), #0 means disable bagging; k means perform bagging at every k iteration
            'seed':7, 
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
    # Predictions are probabilities for each class in this case
    # Recode to what class is predicted (aka has the highes probability)
    import numpy as np
    yhatlgb = [np.argmax(line) for line in ypredlgbm]
    # Results to data frame again
    lgbres = pd.DataFrame({'test':ytest, 'pred':yhatlgb})
    # Recode correct/incorrect predictions
    lgbres['correct'] = 0
    lgbres.loc[lgbres.test == lgbres.pred, 'correct'] = 1
    correct_classes = lgbres['correct'].value_counts()[1]
    # If the number of correct predictions is higher than before -> replace and save parameters of LGB model
    if correct_classes > best_correct_classes:
        best_correct_classes = correct_classes
        best_params = params
        print("Best so far... %s (from 30)" %best_correct_classes)

print("Final results")
print("Best number of correct classes: %s (from 30)" %best_correct_classes)
print(best_params)

# Best number of correct classes: 29 (from 30) = 97%
# {'objective': 'multiclass', 'num_class': 3, 'learning_rate': 5.169297867011159, 'lambda_l1': 1.09327875603221, 'metric': 'multi_logloss', 'feature_fraction': 0.31674309600434786, 'bagging_fraction': 0.6828976377251965, 'bagging_freq': 39, 'seed': 7, 'verbose': -1, 'boosting_type': 'gbdt'}

