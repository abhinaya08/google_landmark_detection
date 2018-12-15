# predicting VGG16

import matplotlib
from keras import backend as K
import matplotlib.pyplot as plt
import argparse
import sys
import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
from multiprocessing import Pool
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils.np_utils import to_categorical
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
#from lenet import LeNet
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
#import imutils
import keras



from tensorflow.python.platform import app


import argparse
import os
import sys
import time
from time import *
import io
import tensorflow as tf


top_model_weights_path = 'bottleneck_fc_model_3.h5'
train_data_dir = "/home/undertakeme/200_classes/train_images/"
testfile = "/home/abhin/test_images/"

subfile = 'vgg16_spezifinal.csv'

img_width, img_height = 128, 128

#train_data_dir = '/home/kevin/LandmarkRec/train_images'

nb_train_samples = 51754
print('finished')
nb_validation_samples = 36062
epochs = 2
batch_size = 10
print ('predict')


print ('starting...')
image_path = testfile
path, dirs, files = next(os.walk(testfile))
file_len = len(files)
print('Number of Testimages:', file_len)

train_datagen = ImageDataGenerator(rescale=1. / 255)

generator = train_datagen.flow_from_directory(train_data_dir, batch_size=batch_size)
label_map = (generator.class_indices)
#print (label_map)

num_classes = 200

# add the path to your test image below

with open(subfile, 'wb') as csvfile:
    newFileWriter = csv.writer(csvfile)
    newFileWriter.writerow(['id', 'landmarks'])

    file_counter = 0
    for root, dirs, files in os.walk(image_path):  # loop through startfolders
        for pic in files:
            t1 = clock()

            #loop folder and convert image
            path = image_path + pic


            orig = cv2.imread(path)
            image = load_img(path, target_size=(128, 128))
            image = img_to_array(image)

            # important! otherwise the predictions will be '0'
            image = image / 255

            image = np.expand_dims(image, axis=0)

            #classify landmark
            base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))


            top_model = Sequential()
            top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(num_classes, activation='softmax'))


            model = Model(input=base_model.input, output=top_model(base_model.output))
            model.load_weights("bottleneck_fc_model_3.h5")

            prediction = model.predict(image)
            #np.savetxt("pred_check.csv", prediction, delimiter=",")
            #print "prediction"
            #print prediction


            class_predicted = prediction.argmax(axis=1)
            #class_predicted = np.argmax(prediction,axis=0)
            #print "class_predicted:"
            #print class_predicted


            inID = class_predicted[0]
            #print inID

            inv_map = {v: k for k, v in label_map.items()}
            #print "inv_map:"
            #print inv_map

            label = inv_map[inID]


            score = max(prediction[0])
            scor = "{:.2f}".format(score)
            out = str(label) + ' '+ scor
            #print (score)



            newFileWriter.writerow([os.path.splitext(pic)[0], out])
            #print (os.path.splitext(pic)[0], out)

            K.clear_session()


#===Testing for one image
from PIL import Image
%matplotlib inline
import numpy as np
from matplotlib.pyplot import imshow
filename = "beach.jpg"
pil_image = Image.open(filename)
pil_image_small = pil_image.resize((128, 128))
pil_image_rgb = pil_image_small.convert('RGB')
imshow(np.asarray(pil_image_rgb))
pil_image_rgb.save("beach.jpg", format='JPEG', quality=90)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))


top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))


model = Model(input=base_model.input, output=top_model(base_model.output))
model.load_weights("bottleneck_fc_model_3.h5")


#orig = cv2.imread("uttower.jpg")
image = load_img("beach.jpg", target_size=(128, 128))
image = img_to_array(image)

# important! otherwise the predictions will be '0'
image = image / 255

image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
class_predicted = prediction.argmax(axis=1)
#class_predicted = np.argmax(prediction,axis=0)
#print "class_predicted:"
#print class_predicted


inID = class_predicted[0]
#print inID

inv_map = {v: k for k, v in label_map.items()}
#print "inv_map:"
#print inv_map

label = inv_map[inID]


score = max(prediction[0])
scor = "{:.2f}".format(score)
out = str(label) + ' '+ scor

print(out)

