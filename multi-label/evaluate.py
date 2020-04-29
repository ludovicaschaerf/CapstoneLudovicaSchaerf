# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:18:06 2020

@author: Ludovica
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from os.path import isfile, join
import pickle
import sys
sys.path.insert(0, '../data')
from datahandler_multilabel import create_dataset
import numpy as np
from sklearn import metrics

def main():

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

    def mean_per_class_accuracy(y, y_hat, num_classes):
        accuracy = [0]*num_classes
        for i in range(num_classes):
            accuracy[i] = 0
            for j in range(len(y)):
                if y[j][i] == y_hat[j][i]:
                    accuracy[i] += 1
            accuracy[i] /= len(y)
        return accuracy
    summary_metrics = {}
    
    list_paths = ['training_flat_multilabel_ResNetfine_tuned.h5', \
                  'training_flat_multilabel_VGGfine_tuned.h5', \
                  'training_flat_multilabel_InceptionV3fine_tuned.h5', \
                  'training_flat_multilabelVGG.h5', \
                  'training_flat_multilabelInceptionV3.h5', \
                  'training_flat_multilabelResNet.h5', \
                 ]
    for model in list_paths:
        model1 = tf.keras.models.load_model('./results/'+model, compile=False)
        print(model)
        model1.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=["binary_accuracy",
                       "categorical_accuracy",
                       precision,
                       recall,
                       macro_f1
                      ])

        with open('../data/train_test_split.pkl', 'rb') as infile:
            train_x, train_y, val_x, val_y, test_x, test_y = pickle.load(infile)
        test_generator = create_dataset(test_x, test_y)
        print(len(test_y[0][:15]))
            
        per_item_metrics = model1.evaluate(test_generator, verbose=2)
        print(per_item_metrics)
        pred = model1.predict_generator(test_generator, verbose=1)
        predicted_class_indices = np.where(pred > 0.5, 1, 0)
        
        for i,label in enumerate(predicted_class_indices):
            predicted_class_indices[i] = label[:15]
            
        per_class_accuracy = mean_per_class_accuracy(
            test_y, predicted_class_indices, len(test_y[0])
        )

        summary_metrics[model] = [metrics.classification_report(test_y, predicted_class_indices, output_dict = True)]
        summary_metrics[model] += per_item_metrics
        print(summary_metrics)
    
    with open('evals.pkl', 'wb') as outfile:
        pickle.dump(summary_metrics, outfile)



if __name__ == '__main__':
    main()
