#====predict Xception


import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
import keras

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 8} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

train_data_dir = "/home/undertakeme/200_classes/train_images"
validation_data_dir = "/home/undertakeme/val_images"

# hyper parameters for model
nb_classes = 200  # number of classes
based_model_last_block_layer_number = 126 
img_width, img_height = 128, 128  # change based on the shape/structure of your images
batch_size = 600 
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
print("THIS CODE IS RUNNING")

base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

# Top Model Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nb_classes, activation='softmax')(x)

# add your top layer block to your base model
model = Model(base_model.input, predictions)
print(model.summary())

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all layers of the based model that is already pre-trained.
for layer in base_model.layers:
    layer.trainable = False

# Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
# To save augmentations un-comment save lines and add to your flow parameters.
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=transformation_ratio,
                                   shear_range=transformation_ratio,
                                   zoom_range=transformation_ratio,
                                   cval=transformation_ratio,
                                   horizontal_flip=True,
                                   vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

#os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
# save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
# save_prefix='aug',
# save_format='jpeg')
# use the above 3 commented lines if you want to save and look at how the data augmentations look like

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

model.compile(optimizer='nadam',
              loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
              metrics=['accuracy'])

# save weights of best training epoch: monitor either val_loss or val_acc

top_weights_path = '/home/undertakeme/200_classes/xception_top_model_weights.h5'
callbacks_list = [
    ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_acc', patience=5, verbose=0)
]

# Train Simple CNN
model.fit_generator(train_generator,
                    steps_per_epoch=51754 //batch_size,
                    nb_epoch=nb_epoch / 5,
                    validation_data=validation_generator,
                    nb_val_samples= 36062 //batch_size,
                    callbacks=callbacks_list)


for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True


model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# save weights of best training epoch: monitor either val_loss or val_acc
final_weights_path = 'model_weights_xception.h5'

callbacks_list = [
    ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, verbose=0)
]

# fine-tune the model
model.fit_generator(train_generator,
                    samples_per_epoch=train_generator.nb_sample,
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=validation_generator.nb_sample,
                    callbacks=callbacks_list)


model_json = model.to_json()
with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
    json_file.write(model_json)


k.clear_session()