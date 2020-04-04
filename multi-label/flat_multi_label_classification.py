# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:18:06 2020

@author: Ludovica
"""
from __future__ import absolute_import, division, print_function, unicode_literals

#new_dataset options: True : the dataset was downloaded recently so I need to filter it
#                     False : the dataset in my computer is the same on filenames.pkl
new_dataset = False

#model_type options: pretrained_no_tuning
#                    pretrained_fine_tuning,
#                    out_of_the_box

model_type = 'pretrained_no_tuning'

#saved options: True : continue training a saved model
#               False : start a new training
saved = False

#pretrained options: VGG
#                    ResNet
#                    InceptionV3
pretrained = 'VGG'

#save_predictions options: True if you wish to already predict on the test set and save it
#                          False otherwise
saved_predictions = False


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

#import modules
from out_of_the_box import Dense, Convolution, MyModel
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
df = shuffle(df,)
print(df.shape, df.columns)

if new_dataset == True:
    mypath = os.path.join('..','..','data_tate')
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    filenames1 = []
    labels1 = []
    for i,img_targ in enumerate(filenames):
        img_targ0 = img_targ.split('\\')[-1]
        if img_targ0 in onlyfiles:
            filenames1.append(str(img_targ))
            labels1.append(labels[i])
    filenames = filenames1
    labels = labels1

df = pd.concat([pd.Series(filenames, name='filenames'), pd.Series(labels, name='labels')], axis=1)
df = shuffle(df, random_state=42)
print(df.shape, df.columns)

#train test val split
train_x = list(df['filenames'][:18000])
train_y = list(df['labels'][:18000])
val_x = list(df['filenames'][18000:21000])
val_y = list(df['labels'][18000:21000])
test_x = list(df['filenames'][21000:])
test_y = list(df['labels'][21000:])
print(len(train_x), len(train_y), len(val_x), len(val_y), len(test_x), len(test_y))

train_generator = create_dataset(train_x, train_y)
val_generator = create_dataset(val_x, val_y)
test_generator = create_dataset(test_x, test_y)

#settings
IMG_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 56
AUTOTUNE = tf.data.experimental.AUTOTUNE

#metrics
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

def mean_per_class_accuracy(y, y_hat):
    y_pred = tf.cast(tf.greater(y_hat, 0.5), tf.float32)
    per_class_acc = []
    per_class_acc.append(tf.cast(tf.math.count_nonzero(y_pred[i] * y[i], axis=0), tf.float32))
    mean_acc = tf.reduce_mean(per_class_acc)
    return mean_acc

def precision(y, y_hat):
    y_pred = tf.cast(tf.greater(y_hat, 0.5), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    pres = tp / (tp + fp + 1e-16)
    precision = tf.reduce_mean(pres)
    return precision

def recall(y, y_hat):
    y_pred = tf.cast(tf.greater(y_hat, 0.5), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    rec = tp / (tp + fn + 1e-16)
    recall = tf.reduce_mean(rec)
    return recall

#models

feature_extractor_layer = {}
feature_extractor_layer['VGG'] = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
feature_extractor_layer['ResNet'] = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
feature_extractor_layer['InceptionV3'] = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))


if model_type == 'pretrained_no_tuning':
    feature_extractor_layer[pretrained].trainable = False
    print(feature_extractor_layer[pretrained].summary())

    csv_logger = tf.keras.callbacks.CSVLogger('training_flat_multilabel_'+str(pretrained)+'.csv')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        'training_flat_multilabel'+str(pretrained)+'.h5', save_best_only=True,
                )

    if saved == False:
        model = tf.keras.Sequential([
            feature_extractor_layer[pretrained],
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation='relu', name='hidden_layer'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(15, activation='sigmoid', name='output')
        ])
    #in alternative, if you wish to resume training:
    else:
        model = tf.keras.models.load_model('training_flat_multilabel'+str(pretrained)+'.h5')

    print(model.summary())

if model_type == 'pretrained_fine_tuning':
    feature_extractor_layer[pretrained].trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 6*len(feature_extractor_layer[pretrained].layers)//7

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in feature_extractor_layer[pretrained].layers[:fine_tune_at]:
        layer.trainable =  False

    print(feature_extractor_layer[pretrained].summary())

    if saved == False:
        model = tf.keras.Sequential([
            feature_extractor_layer[pretrained],
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation='relu', name='hidden_layer'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(15, activation='sigmoid', name='output')
        ])
        print(model.summary())
    else:
        #in alternative, if you wish to resume training:
        model = tf.keras.models.load_model('training_flat_multilabel_'+str(pretrained)+'fine_tuned.h5')

    csv_logger = tf.keras.callbacks.CSVLogger('training_flat_multilabel_'+str(pretrained)+'fine_tuned.csv')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        'training_flat_multilabel_'+str(pretrained)+'fine_tuned.h5', save_best_only=True,
                )
if model_type == 'out_of_the_box':
    if saved == False:
        model = MyModel([64, 64, 128, 128],[128, 64, 15])
        model.build(input_shape=(BATCH_SIZE, 224, 224, 3))
        print(model.summary())
    else:
        #in alternative, if you wish to resume training:
        model = tf.keras.models.load_model('training_flat_multilabel_out_the_box.h5')

    csv_logger = tf.keras.callbacks.CSVLogger('training_flat_multilabel_out_the_box.csv')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        'training_flat_multilabel_out_the_box.h5', save_best_only=True,
                )



model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=["binary_accuracy",
                       "categorical_accuracy",
                       precision,
                       recall,
                       macro_f1
                       ])


model.fit(
    train_generator,
    epochs=30,
    validation_data = val_generator,
    callbacks = [csv_logger, checkpoint],
)

if saved_predictions:
    pred = model.predict_generator(test_generator, verbose=1)
    predicted_class_indices = np.where(pred > 0.5, 1, 0)
    labels_dict = {0:'people',1:'objects',2:'places',3:'architecture',4:'abstraction',5:'society',\
          6:'nature',7:'emotions, concepts and ideas',8:'interiors',9:'work and occupations', \
          10:'symbols & personifications',11:'religion and belief',12:'leisure and pastimes',\
          13:'history',14:'literature and fiction',15:'group/movement'}
    predictions = [[]]*len(predicted_class_indices)
    actual = [[]]*len(test_y)
    for k in range(len(predicted_class_indices)):
        predictions[k] = []
        actual[k] = []
        for i in range(len(predicted_class_indices[k])):
            if predicted_class_indices[k][i] == 1:
                predictions[k].append(labels_dict[i])
            if test_y[k][i] == 1:
                actual[k].append(labels_dict[i])
                
    print(len(predictions), len(actual))
    results=pd.DataFrame({"Filename":test_x,
                          "Actual":actual,
                          "Predictions":predictions})
    results.to_csv("results_"+pretrained+model_type+".csv",index=False)