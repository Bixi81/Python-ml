# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb

from keras.applications import VGG16
import os, datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import models, layers, optimizers, regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from PIL import ImageFile
import statistics
ImageFile.LOAD_TRUNCATED_IMAGES = True

###############################################
# DIR with training images
base_dir = 'C:/kerasimages'
# Number training images
ntrain = 2000
# Number validation images
nval  = 500
# Batch size
batch_size = 20 #20
# Epochs (fine tuning [100])
ep = 400 #400
# Epochs (first step [30])
ep_first = 30 
# Number of classes (for training, output layer)
nclasses = 30
###############################################
start = datetime.datetime.now()

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
#test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)

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
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, ntrain)
validation_features, validation_labels = extract_features(validation_dir, nval)
#test_features, test_labels = extract_features(test_dir, 1000)

# Labels and features
train_labels = to_categorical(train_labels)
validation_labels = to_categorical(validation_labels)
#test_labels = to_categorical(test_labels)
train_features = np.reshape(train_features, (ntrain, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (nval, 4 * 4 * 512))
#test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

#######################################
# Model
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(BatchNormalization())

model.add(layers.Dense(2048, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(layers.Dense(2048, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(BatchNormalization())

model.add(layers.Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(layers.Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(BatchNormalization())

model.add(layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(BatchNormalization())

model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(BatchNormalization())

model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002
model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.003)))#0.002

model.add(layers.Dense(nclasses, activation='softmax'))
conv_base.trainable = False

#######################################
# Data generators
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

# Model compile / fit
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

# early stopping: https://keras.io/callbacks/#earlystopping
es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, verbose=1, patience=40, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.9, patience=15, min_lr=1e-20, verbose=1, cooldown=3)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=round((ntrain+nval)/batch_size,0),
      epochs=ep_first,
      validation_data=validation_generator,
      validation_steps=20, #50
      verbose=2,
      callbacks=[es, reduce_lr])

#######################################
# Fine tuning
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.00001), #1e-5
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=round((ntrain+nval)/batch_size,0),
      epochs=ep,
      validation_data=validation_generator,
      validation_steps=20,
      callbacks=[es, reduce_lr])

#######################################
# Save model
model.save('C:/Users/User/Documents/Python/image_multiclass_keras/keras_multiclass_flickr_model.hdf5')
end = datetime.datetime.now()
delta = str(end-start)

# Metrics
acc = history.history['acc']
acc = acc[-5:]
val_acc = history.history['val_acc']
val_acc = val_acc[-5:]
loss = history.history['loss']
loss = loss[-5:]
val_loss = history.history['val_loss']
val_loss = val_loss[-5:]

# End statement
print("============================================")
print("Time taken (h/m/s): %s" %delta[:7])
print("============================================")
print("Metrics (average last five steps)")
print("--------------------------------------------")
print("Loss       %.3f" %statistics.mean(loss))
print("Val. Loss  %.3f" %statistics.mean(val_loss))
print("--------------------------------------------")
print("Acc.       %.3f" %statistics.mean(acc))
print("Val. Acc.  %.3f" %statistics.mean(val_acc))
print("============================================")
print("Epochs:    %s / %s" %(ep,ep_first))
