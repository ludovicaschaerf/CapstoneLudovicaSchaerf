# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:18:06 2020

@author: Ludovica
"""
from __future__ import absolute_import, division, print_function, unicode_literals

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
import argparse

import os
from os import listdir
from os.path import isfile, join
import pickle

#import modules
from out_of_the_box import Dense, Convolution, MyModel
import sys
sys.path.insert(0, '../data')
from datahandler_multilabel import create_dataset
from sklearn.utils import class_weight



# to fix some warnings
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 999999999999 # Fix DecompressionBombWarning

def main(args):
    IMG_SIZE = 224
    CHANNELS = 3
    
    with open('../data/filenames.pkl', 'rb') as infile:
        filenames = pickle.load(infile)
    
    with open('../data/labels.pkl', 'rb') as infile2:
        labels = pickle.load(infile2)
    
    df = pd.concat([pd.Series(filenames, name='filenames'), pd.Series(labels, name='labels')], axis=1)
    df = shuffle(df,)
    print(df.shape, df.columns)
    if args.new_dataset == True:
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
    
    train_generator = create_dataset(train_x, train_y, BATCH_SIZE=args.batch_size)
    val_generator = create_dataset(val_x, val_y, BATCH_SIZE=args.batch_size)
    test_generator = create_dataset(test_x, test_y, BATCH_SIZE=args.batch_size)
    
    
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
    
    # class_weights
    num_elts = [14452, 9418, 6665, 8159, 7225, 7371, 12137, 6101, 1774, 4047, 2360, 1998, 2218, 1205, 1960]
    total = sum(num_elts)
    print(total)
    if args.class_weights:
        class_weight = {0: total/num_elts[0], 1: total/num_elts[1], \
                        2: total/num_elts[2], 3: total/num_elts[3], \
                        4: total/num_elts[4], 5: total/num_elts[5], \
                        6: total/num_elts[6], 7: total/num_elts[7], \
                        8: total/num_elts[8], 9: total/num_elts[9], \
                        10: total/num_elts[10], 11: total/num_elts[11], \
                        12: total/num_elts[12], 13: total/num_elts[13], \
                        14: total/num_elts[14]}
    else:
        class_weight = None
    
    #models    
    feature_extractor_layer = {}
    feature_extractor_layer['VGG'] = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', \
                                                                       input_shape= (IMG_SIZE,IMG_SIZE,CHANNELS))
    feature_extractor_layer['ResNet'] = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',\
                                                                              input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
    feature_extractor_layer['InceptionV3'] = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', \
                                                                                         input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
    
    
    if args.model_type == 'pretrained_no_tuning':
        feature_extractor_layer[args.pretrained].trainable = False
        print(feature_extractor_layer[args.pretrained].summary())
    
        csv_logger = tf.keras.callbacks.CSVLogger('./results/training_flat_multilabel_'+str(args.pretrained)+'.csv')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            './results/training_flat_multilabel'+str(args.pretrained)+'.h5', save_best_only=True,
                    )
    
        if args.saved == False:
            model = tf.keras.Sequential([
                feature_extractor_layer[args.pretrained],
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1024, activation='relu', name='hidden_layer'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(15, activation='sigmoid', name='output')
            ])
        #in alternative, if you wish to resume training:
        else:
            model = tf.keras.models.load_model('./results/training_flat_multilabel'+str(args.pretrained)+'.h5')
    
        print(model.summary())
    
    if args.model_type == 'pretrained_fine_tuning':
        feature_extractor_layer[args.pretrained].trainable = True
    
        # Fine-tune from this layer onwards
        fine_tune_at = 6*len(feature_extractor_layer[args.pretrained].layers)//7
    
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in feature_extractor_layer[args.pretrained].layers[:fine_tune_at]:
            layer.trainable =  False
    
        print(feature_extractor_layer[args.pretrained].summary())
    
        if args.saved == False:
            model = tf.keras.Sequential([
                feature_extractor_layer[args.pretrained],
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1024, activation='relu', name='hidden_layer'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(15, activation='sigmoid', name='output')
            ])
            print(model.summary())
        else:
            #in alternative, if you wish to resume training:
            model = tf.keras.models.load_model('./results/training_flat_multilabel_'+str(args.pretrained)+'fine_tuned.h5')
    
        csv_logger = tf.keras.callbacks.CSVLogger('./results/training_flat_multilabel_'+str(args.pretrained)+'fine_tuned.csv')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            './results/training_flat_multilabel_'+str(args.pretrained)+'fine_tuned.h5', save_best_only=True,
                    )
    if args.model_type == 'out_of_the_box':
        if args.saved == False:
            model = MyModel([64, 64, 128, 128],[128, 64, 15])
            model.build(input_shape=(args.batch_size, 224, 224, 3))
            print(model.summary())
        else:
            #in alternative, if you wish to resume training:
            model = tf.keras.models.load_model('./results/training_flat_multilabel_out_the_box.h5')
    
        csv_logger = tf.keras.callbacks.CSVLogger('./results/training_flat_multilabel_out_the_box.csv')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            './results/training_flat_multilabel_out_the_box.h5', save_best_only=True,
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
        epochs=args.num_epochs,
        validation_data = val_generator,
        callbacks = [csv_logger, checkpoint],
        workers = args.num_workers,
        class_weight = class_weight
    )
    
    if args.saved_predictions:
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
        results.to_csv("./results/results_"+pretrained+model_type+".csv",index=False)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_dataset', default=False , help='True : the dataset was downloaded recently so I need to filter it, False : the dataset in my computer is the same on filenames.pkl')
    parser.add_argument('--model_type', type=str, default='pretrained_no_tuning' , help='options pretrained_no_tuning, pretrained_fine_tuning, out_of_the_box')
    parser.add_argument('--saved', default=False, help='True : continue training a saved model, False : start a new training')
    parser.add_argument('--pretrained', type=str, default='VGG', help='VGG, ResNet, InceptionV3')
    parser.add_argument('--saved_predictions', default=False, help='True if you wish to already predict on the test set and save it, False otherwise')
    parser.add_argument('--class_weights', default=True, help='True if you wish to use class weights for a balanced classification, False otherwise')
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=56)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)