#classify xception

import sys
import os
import csv
import glob
import shutil
import numpy as np
import tensorflow as tf
#from keras.utils.np_utils import probas_to_classes
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.models import model_from_json
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# parameters dependent on your dataset: modified to your example
img_width, img_height = 128, 128  # must match the fix size of your train image sizes. 600, 150 for text_images
batch_size = 600  

# default paths
model_name = '/home/undertakeme/200_classes/model.json'
model_weights = '/home/undertakeme/200_classes/xception_model_weights.h5'
results_name = 'xception_predictions.csv'
test_data_dir = "/home/abhin/test_images/"
train_data_dir = "/home/undertakeme/200_classes/train_images/"
val_data_dir = "/home/undertakeme/val_images"

json_file = open(model_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_weights)


#==============predicting for validation images
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(val_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  shuffle=False)

# Calculate class posteriors probabilities
y_probabilities = model.predict_generator(val_generator,
                                          val_samples=val_generator.nb_sample)
# Calculate class labels
y_classes = probas_to_classes(y_probabilities)
filenames = [filename.split('/')[1] for filename in test_generator.filenames]
ids = [filename.split('.')[0] for filename in filenames]

# save results as a csv file in the specified results directory
with open(os.path.join(results_path, results_name), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(('id', 'class0_prob', 'class1_prob', 'label'))
    writer.writerows(zip(ids, y_probabilities[:, 0], y_probabilities[:, 1], y_classes))



#=============Predicting for test images
subfile = 'xception_spezifinal.csv'
with open(subfile, 'wb') as csvfile:
    newFileWriter = csv.writer(csvfile)
    newFileWriter.writerow(['id', 'landmarks'])

    file_counter = 0

    for root, dirs, files in os.walk(test_data_dir):  # loop through startfolders
            for pic in files:
                #loop folder and convert image
                path = test_data_dir + pic

                orig = cv2.imread(path)
                image = load_img(path, target_size=(128, 128))
                image = img_to_array(image)

                # important! otherwise the predictions will be '0'
                image = image / 255

                image = np.expand_dims(image, axis=0)
                prediction = model.predict(image)
                #print(prediction)
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

                k.clear_session()