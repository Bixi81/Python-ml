import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import catboost as cb
from catboost import Pool, CatBoostRegressor
from bayes_opt import BayesianOptimization

# Load data
dataframe = pd.read_csv("C:/Users/User/.../data.csv", sep=";")

# Test/train split
msk = np.random.rand(len(dataframe)) < 0.8
train = dataframe[msk]
test = dataframe[~msk]

# Define x,y
xtrain = train.drop(['ffdelta'], axis=1)
ytrain = train[['ffdelta']]
xtest = test.drop(['ffdelta'], axis=1)
ytest = test[['ffdelta']]

# Collect categocial features 
i = 0
featurelist = []
for column in xtrain:
    if xtrain[column].min()==0  and xtrain[column].max()==1:
        featurelist.append(i)
    i=i+1

# Define: Boosting rounds, early stopping rounds
brounds = 999999
esrounds = 100

# Parse list of categorical features
cat_features = featurelist

# Train data
train_dataset = Pool(data=xtrain,
                     label=ytrain,
                     cat_features=cat_features)

# Test data
eval_dataset = Pool(data=xtest,
                    label=ytest,
                    cat_features=cat_features)

# MODEL
# Catboost docs: https://catboost.ai/docs/concepts/python-reference_catboost_fit.html

# Function is used for optimisation, so arguments in function must match arguments in params and pbounds below
def cbfunc(border_count,l2_leaf_reg, depth, learning_rate):
    params = {
        'eval_metric':'MAE', # using MAE here, could also be RMSE or MSE
        'early_stopping_rounds': esrounds,
        'num_boost_round': brounds,
        'use_best_model': True,
        'task_type': "GPU"
    }

    params['border_count'] = round(border_count,0)
    params['l2_leaf_reg'] = l2_leaf_reg
    params['depth'] = round(depth,0)
    params['learning_rate'] = learning_rate

    # Cross validation   
    cv_results = cb.cv(cb.Pool(xtrain, ytrain, cat_features=cat_features), params=params,fold_count=3,inverted=False,partition_random_seed=5,shuffle=True, logging_level='Silent') 
    # bayes_opt MAXIMISES: In order to minimise MAE, I use 1/MAE as target value
    return 1/cv_results['test-MAE-mean'].min()

pbounds = { 
        'border_count': (1,255),      # int. 1-255
        'l2_leaf_reg': (0,20),        # any positive value
        'depth': (1,16),              # int. up to  16
        'learning_rate': (0.01,0.2),
    }

optimizer = BayesianOptimization(
    f=cbfunc,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=5
)

optimizer.maximize(
    init_points=2,
    n_iter=500,
)

print("list of results")
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("best result")
print(optimizer.max)
