from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import random

import keras
import keras.utils
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras import optimizers

##########################################
# Load iris data as pandas df
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Stratified train/test split -> gives a balanced sample
train, test = train_test_split(df, test_size=0.2, stratify=df['target']) #random_state=2021

### SCALE DATA
# Set up scaler - make sure scaler is based on train data
scaler = StandardScaler().fit(train.iloc[:, 0:4])
# Scale train data
train.iloc[:, 0:4] = scaler.transform(train.iloc[:, 0:4])
# Scale test data
test.iloc[:, 0:4] = scaler.transform(test.iloc[:, 0:4])

##########################################
# FUNCTIONS

# Assign training data
# For n times...
# Randomly pick two rows from same class -> append to np with label "same"
# Randomly pick one row from other class -> append one row from previous step and row from other class with label "different"

# Function to generate train / test pairs
def makepairs(dataframe,npairs):
    # Shape: Number pairs, 2 (as a pair), one row per pair, four variables (columns) per pair
    x_arr = np.zeros([int(npairs), 2, 1, 4]) 
    # Shape: Number pairs, one label per pair
    y_arr = np.zeros([int(npairs), 1]) 
    # For each class one "same" and one "different" pair is made
    # Adjust npairs => npairs = ( npairs / 2 ) / len(set(dataframe['target']))
    nn = 0
    while nn < npairs:
        # One pair true/false for each class
        for c in set(dataframe['target']):
            ## SAME CLASS
            try:
                # Append random rows from class (and drop target)
                x_arr[nn, 0, :] = dataframe[dataframe['target']==c].sample(n=1).drop(['target'], axis=1).values
                x_arr[nn, 1, :] = dataframe[dataframe['target']==c].sample(n=1).drop(['target'], axis=1).values
                # Append labels = 1 for "is same class"
                y_arr[nn] = 1
                nn = nn + 1
            except:
                pass
            ## DIFFERENT CLASS
            # Randomly pick different class
            try:
                while True:
                    cl = list(set(dataframe['target']))
                    rclass = random.choice(cl)
                    if rclass!=c:
                        break
                # Append random row to np array (make sure different classes are chosen)
                x_arr[nn, 0, :] = dataframe[dataframe['target']==c].sample(n=1).drop(['target'], axis=1).values
                x_arr[nn, 1, :] = dataframe[dataframe['target']==rclass].sample(n=1).drop(['target'], axis=1).values
                # Append labels = 0 for "is different class"
                y_arr[nn] = 0
                nn = nn + 1
            except:
                pass
    return x_arr, y_arr

# Function to predict rows (X_i) -> takes pandas df with cases to predict
# Compares unknown cases (in pred_dataframe) with known cases (comp_dataframe)
def predictpairs(pred_dataframe, comp_dataframe, npairs):
    ### X contains X_UNKOWN <- paired -> X_KNOWN  |  y contains LABEL_KNOWN
    #################################
    # Initiate np array -> length = nrows(pred_df) * classes(comp_df) * npairs
    nplength = pred_dataframe.shape[0] * len(set(comp_dataframe['target'])) * npairs
    # Shape: Number pairs, 2 (as a pair), one row per pair, four variables (columns) per pair
    x_arr = np.zeros([int(nplength), 2, 1, 4]) 
    # Shape: Number pairs, one label per pair
    y_arr = np.zeros([int(nplength), 1]) 
    #################################
    # Drop target in pred_df is exists
    try:
        pred_dataframe = pred_dataframe.drop(['target'], axis=1)
    except:
        pass
    # For each row in pred_df
    nn = 0
    for index, row in pred_dataframe.iterrows():
        # Generate n paris for each label in comp_df
        for c in set(comp_dataframe['target']):
            for n in range(npairs):
                ## GENERATE A PAIR
                # Append rows from pred_df to be tested
                x_arr[nn, 0, :] = row.values
                # Append random row from known class in comp_df
                x_arr[nn, 1, :] = comp_dataframe[comp_dataframe['target']==c].sample(n=1).drop(['target'], axis=1).values
                # Append class labels (from comp_df)
                y_arr[nn] = c
                # Row counter
                nn = nn + 1
    return x_arr, y_arr

# Takes single row from test data to predict (by comparison to npairs obs. from comp_df)
def pred_row(rowtopredict, comp_dataframe, npairs):
    try:
        test_target = rowtopredict["target"]
    except:
        pass
    try:
        test_x = rowtopredict.drop(['target'], axis=1)
    except:
        test_x = rowtopredict
    # Get pairs to compare with true target(s)
    xpairs, comp_target = predictpairs(test_x, comp_dataframe, npairs)
    return xpairs, comp_target, test_target

# Generate pairs for learning
xtrain, ytrain = makepairs(train,50)
xtest, ytest = makepairs(test,50)

##########################################
### MODEL
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto', restore_best_weights=True)

def build_base_network(input_shape):
    seq = Sequential()
    seq.add(Dense(32, activation='relu',input_shape=input_shape, kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)))
    #seq.add(Dropout(0.05))
    #seq.add(Dense(32, activation='relu'))
    #seq.add(Dropout(0.05))
    seq.add(Dense(8, activation='relu'))
    #seq.add(Dropout(0.05))
    seq.add(Dense(1, activation='relu'))
    seq.add(Flatten())
    return seq

# Pairs to be fed to model
input_dim = xtrain.shape[2:]
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)

###

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Define model and eucl. distance
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
model = Model(input=[img_a, img_b], output=distance)

## Train
model.compile(loss=contrastive_loss, optimizer=keras.optimizers.RMSprop(lr=0.001))
img_1 = xtrain[:, 0]
img_2 = xtrain[:, 1] 

model.summary()

model.fit([img_1, img_2], ytrain, 
            validation_split=.2, 
            batch_size=32, 
            verbose=1, 
            epochs=300,
            callbacks=[early_stopping])

##########################################
### Predict test data
pred = model.predict([xtest[:,0], xtest[:,1]])
# Pred to DF with labels
result = pd.DataFrame(data=pred)
result['true']=ytest
result.columns = ['dist', 'true']
#print(result)

### Predict some data line by line and compare similarity to p observations with known class
for p in [20,100]:
    true=0
    false=0
    total=0
    for r in range(0,25):
        # Predict many pairs for ONE test observation
        xpairs, truelabel, testlabel = pred_row(test.iloc[[r]], train, p)
        pred3 = model.predict([xpairs[:,0], xpairs[:,1]])
        result3 = pd.DataFrame(data=pred3)
        result3['label']=truelabel
        result3.columns = ['dist', 'label']
        agg = result3.groupby(['label'], as_index=False).mean() #.sort_values('dist')
        predicted_class = agg.loc[agg['dist'].idxmin()]['label']
        total = total+1
        if testlabel.values[0] == predicted_class:
            true = true+1
        else:
            false = false+1

    print(true," out of ", total, " -> ", true/total, " | pairs = ",p)
