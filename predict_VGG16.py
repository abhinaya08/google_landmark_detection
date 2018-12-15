#== VGG-16 model

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import csv
import os
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import math
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
#import imutils
import tensorflow as tf
import keras
#from lenet import LeNet
#from imutils import paths

config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 24} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

img_width, img_height = 128, 128

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = "/home/undertakeme/200_classes/train_images"
val_data_dir = "/home/undertakeme/val_images"
top_model_weights_path_1 = 'bottleneck_fc_model_2.h5'

#nb_train_samples = count(train_data_dir)
#nb_validation_samples = count(validation_data_dir)
epochs = 3
batch_size = 600


datagen = ImageDataGenerator(rescale=1. / 255)
#datagen.
# build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(128,128,3))

generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)

bottleneck_features_train = model.predict_generator(generator, 51754)
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)
bottleneck_features_validation = model.predict_generator(generator, 36062)
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

validation_data = np.load(open('bottleneck_features_validation.npy'))
train_data = np.load(open('bottleneck_features_train.npy'))
print(validation_data.shape)
print(train_data.shape)

train_labels = np.array([])
for i in range(200):
    current = np.array([i] * (train_data.shape[0]/200)).astype(int)
    train_labels = np.append(train_labels, current)
#train_labels = to_categorical(train_labels)

validation_labels = np.array([])
for i in range(200):
    current = np.array([i] * (validation_data.shape[0]/200)).astype(int)
    validation_labels = np.append(validation_labels, current)
#validation_labels = to_categorical(validation_labels)

print(train_labels.shape)
print(validation_labels.shape)

# train_data = np.load(open('bottleneck_features_train.npy'))
# train_data = bottleneck_features_train
# train_labels = np.array([0] * (51754 / 2) + [1] * (51754 / 2))

# validation_data = np.load(open('bottleneck_features_validation.npy'))
# validation_labels = np.array(
#     [0] * (36062 / 2) + [1] * (36062 / 2))
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train_labels = to_categorical(train_labels, 99)
#validation_labels = to_categorical(validation_labels, 99)
# callbacks_list = [
#     ModelCheckpoint(top_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
#     EarlyStopping(monitor='val_acc', patience=5, verbose=0)
# ]

model.fit(train_data, train_labels,
          epochs=50,
          batch_size=600,
          validation_data=(validation_data, validation_labels))
model.save_weights(top_model_weights_path)

#import h5py
base_model = applications.VGG16(weights='imagenet',include_top= False,input_shape=(128,128,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(200, activation='softmax'))


model_load = Model(input= base_model.input, output= top_model(base_model.output))

#f = h5py.File(top_model_weights_path)

model_load.load_weights(top_model_weights_path, by_name=True)
for layer in model_load.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model_load.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

np.save('class_indices.npy', train_generator.class_indices)

#after initializing VGG16, load top model weights
#model.load_weights(top_model_weights_path)
top_model_weights_path_1 = "bottleneck_fc_model_4.h5"
model_load.load_weights(top_model_weights_path_1)

#top_model_weights_path_1 = "bottleneck_fc_model_2.h5"
#freeze first 15 layers
validation_generator = test_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

model_load.fit_generator(
    train_generator,
    steps_per_epoch=10000,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=100,
    callbacks=[ModelCheckpoint(filepath="bottleneck_fc_model_5.h5", save_best_only=True,
                              save_weights_only=True)])

