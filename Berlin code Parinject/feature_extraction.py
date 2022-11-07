# -*- coding: utf-8 -*-

import tensorflow as tf
from numpy import array
from os import listdir
from pickle import dump
from pickle import load
import numpy as np
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def extract_features(directory):
    model=tf.keras.applications.VGG16()
    model.layers.pop()
    model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-1].output)
#     DNet=resnet152.ResNet152(include_top=True, weights='imagenet',input_shape=(224,224,3))
#     DNet.layers.pop()
#     print(DNet.summary())
#     DNet = Model(inputs=DNet.inputs, outputs=DNet.layers[-1].output)
    #print(DNet.summary())
    features=dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        #image=np.expand_dims(image,axis=0)
        image = tf.keras.applications.vgg16.preprocess_input(image)
        feature = model.predict(image, verbose=0)
        print (len(feature[0]))
        image_id = name.split('.')[0]
        features[image_id] = feature
        print('>%s' % name)
    return features
# extract features from all images
#directory = 'F:/Users/Genci/Documents/Captioning datasets/UAV/imgs'

directory = 'imgs' 
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features/uav_vgg16_UCM_features.pkl', 'wb')) 

