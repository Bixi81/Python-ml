import numpy as np
import pandas as pd
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

# Standardise data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# Add an additional target (just add some random noise to the original one)
import random
train_targets2 = train_targets + random.uniform(0, 0.1)
test_targets2   = test_targets + random.uniform(0, 0.1)

# https://keras.io/models/model/
from keras import models
from keras import layers
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization

# Input and model architecture
Input_1=Input(shape=(13, ))
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.05))(Input_1)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.05))(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.05))(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.05))(x)
x = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.05))(x)

# Outputs
out1 = Dense(1)(x)
out2 = Dense(1)(x)

# Compile/fit the model
model = Model(inputs=Input_1, outputs=[out1,out2])
model.compile(optimizer = "rmsprop", loss = 'mse')
# Add actual data here in the fit statement
model.fit(train_data, [train_targets,train_targets2], epochs=500, batch_size=4, verbose=0, validation_split=0.2)

# Predict / check type and shape
preds = np.array(model.predict(test_data))
#print(type(preds), preds.shape)
# is a 3D numpy array

# get first part of prediction (column/row/3D layer)
preds0 = preds[0,:,0]
# second part
preds1 = preds[1,:,0]

# Check MAE
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(test_targets, preds0))
print(mean_absolute_error(test_targets2, preds1))
