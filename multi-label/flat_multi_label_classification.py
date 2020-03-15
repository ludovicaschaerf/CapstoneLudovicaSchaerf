# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:18:06 2020

@author: Ludovica
"""

from __future__ import absolute_import, division, print_function, unicode_literals

#! pip install tensorflow
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# import norm
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display


import os
from os import listdir
from os.path import isfile, join
import pickle
#importing module
import sys
sys.path.insert(0, '../data')
from datahandler_multilabel import create_dataset


# to fix some warnings
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 999999999999 # Fix DecompressionBombWarning

with open('../data/filenames.pkl', 'rb') as infile:
    filenames = pickle.load(infile)
    
with open('../data/labels.pkl', 'rb') as infile2:
    labels = pickle.load(infile2)
    
df = pd.concat([pd.Series(filenames, name='filenames'), pd.Series(labels, name='labels')], axis=1)
df = shuffle(df)
print(df.shape, df.columns)

mypath = os.path.join('..','..','data_tate')
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] 

filenames1 = []
labels1 = []
for i,img_targ in enumerate(filenames):
    img_targ0 = img_targ.split('\\')[-1]
    if img_targ0 in onlyfiles:
        filenames1.append(str(img_targ))
        labels1.append(labels[i]) 
        
df = pd.concat([pd.Series(filenames1, name='filenames'), pd.Series(labels1, name='labels')], axis=1)
df = shuffle(df)
print(df.shape, df.columns)

#train test split
train_x = list(df['filenames'][:18000])
train_y = list(df['labels'][:18000])
test_x = list(df['filenames'][18000:])
test_y = list(df['labels'][18000:])
print(len(train_x), len(train_y))

train_generator = create_dataset(train_x, train_y)
val_generator = create_dataset(test_x, test_y)

IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model
BATCH_SIZE = 256 # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU idle time
SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations
STEPS_PER_EPOCH = np.ceil(18000/ BATCH_SIZE)

feature_extractor_layer = {}
# VGG
feature_extractor_layer['VGG'] = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
feature_extractor_layer['ResNet'] = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
feature_extractor_layer['InceptionV3'] = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))

pretrained = 'VGG'
feature_extractor_layer[pretrained].trainable = False
print(feature_extractor_layer[pretrained].summary())

def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

csv_logger = tf.keras.callbacks.CSVLogger('training_flat_multilabel_'+str(pretrained)+'.csv')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    'training_flat_multilabel'+str(pretrained)+'.h5', save_best_only=True,
            )

#if you wish to train a new model:
model = tf.keras.Sequential([
    feature_extractor_layer[pretrained],
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu', name='hidden_layer'),
    tf.keras.layers.Dense(16, activation='sigmoid', name='output')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=["accuracy",
                       "binary_accuracy",
                       "categorical_accuracy",
                       macro_f1])
#in alternative, if you wish to resume training:
#model = tf.keras.models.load_model('training_flat_multilabel'+str(pretrained)+'.h5')
    
print(model.summary())

model.fit(
        train_generator,
        epochs=30,
        validation_data = val_generator,
        callbacks = [csv_logger, checkpoint],
)