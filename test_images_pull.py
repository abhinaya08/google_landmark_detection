
import numpy as np 
import pandas as pd 
import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO


data=pd.read_csv('test.csv')
links=data['url']
id=data['id']
i=-1

print("##############################Starting test images##########################################")
# seperate images into follwing format
# folder name = landmark_id, group images with same landmark id into same folder
# image name = id.jpg
for link in links:              #looping over links to get images
    i+=1
    dir_path = "test_images/"
    filename = dir_path + '/' +  str(id[i]) + '.jpg'
    if os.path.exists(filename):
        #print('Image {} already exists. Skipping download.'.format(filename))
        next
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path)
    #fetch_image(link, dir_path)
    #filename = os.path.join(dir_path, '{}.jpg'.format(key))
    try:
        response = request.urlopen(link)
        image_data = response.read()
    except:
        print('Warning: Could not download image')
        next
    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image')
        next
    try:
        pil_image_small = pil_image.resize((128, 128))
        pil_image_rgb = pil_image_small.convert('RGB')
    except:
        print('Warning: Failed to convert image to RGB')
        next
    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
        #os.rename(dir_path + '/image.jpg', dir_path + '/' +  str(id[i]) + '.jpg')
        #print('Image saved')
    except:
        print('Warning: Failed to save image {}'.format(filename))
        next