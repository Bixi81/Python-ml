# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb

import keras, glob
from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

# Dir with images
base_dir = 'C:/images'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
datagen = ImageDataGenerator(rescale=1./255)

#################################################
batch_size = 20

# Epochs 
firstep  = 50
secondep = 4000

# Number of images
ntrain = 60000
nval   = 60000

# Learning  rate(s)
lr_first  = 1e-5
lr_second = 0.0000008

# Dropout
dro = 0.06

# Early stopping
early_stopping1 = EarlyStopping(monitor='loss', patience=5, mode='auto', restore_best_weights=True)
early_stopping2 = EarlyStopping(monitor='loss', patience=100, mode='auto', restore_best_weights=True)

#################################################
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, ntrain)
validation_features, validation_labels = extract_features(validation_dir, nval)

train_features = np.reshape(train_features, (ntrain, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (nval, 4 * 4 * 512))

#############################################
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout
from keras import regularizers
#from keras.layers import Dropout

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())

model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dro))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
conv_base.trainable = False

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr_first), # 2e-5
              metrics=['acc'])

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=firstep,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2,
      callbacks=[early_stopping1])

#######################################
# Fine tuning

#conv_base.summary()
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr_second), # 1e-6
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=secondep,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks=[early_stopping2])

model.save('C:/pathtomodel/mymodel.hdf5')

#######################################
# Plot
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
