from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


class Convolution(tf.keras.layers.Layer):
    def __init__(self, filters, shape=(224, 224, 3), kernel_size=3,
                 pool_size=(2,2), activation='relu', padding='same'):
        super(Convolution, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape = shape,
        )
        self.conv_layer = tf.keras.layers.Conv2D(
            filters = filters[0], 
            kernel_size = kernel_size,
            activation = activation,
            )
        self.nn_pooling = tf.keras.layers.MaxPooling2D(
            pool_size = pool_size,
            padding = padding,
            )
        self.nn_dropout = tf.keras.layers.Dropout(0.2)
        self.nn_batchnorm = tf.keras.layers.BatchNormalization(axis=-1)
        self.nn_batchnorm1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.nn_batchnorm2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.nn_batchnorm3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv_layer1 = tf.keras.layers.Conv2D(
            filters = filters[1], 
            kernel_size = kernel_size,
            activation = activation,
            )
        self.conv_layer2 = tf.keras.layers.Conv2D(
            filters = filters[2], 
            kernel_size = kernel_size,
            activation = activation,
            )
        self.conv_layer3 = tf.keras.layers.Conv2D(
            filters = filters[3], 
            kernel_size = kernel_size,
            activation = activation,
            )
        
        
    def call(self, input_features):
        activation = self.input_layer(input_features)
        activation = self.conv_layer(activation)
        activation = self.nn_batchnorm(activation)
        activation = self.nn_pooling(activation)
        #activation = self.nn_dropout(activation)
        activation = self.conv_layer1(activation)
        activation = self.nn_batchnorm1(activation)
        activation = self.nn_pooling(activation)
        #activation = self.nn_dropout(activation)
        activation = self.conv_layer2(activation)
        activation = self.nn_batchnorm2(activation)
        activation = self.nn_pooling(activation)
        #activation = self.nn_dropout(activation)
        activation = self.conv_layer3(activation)
        activation = self.nn_batchnorm3(activation)
        activation = self.nn_pooling(activation)
        return self.nn_dropout(activation)

    
class Dense(tf.keras.layers.Layer):
    def __init__(self, nodes, activation='relu'):
        super(Dense, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(
            nodes[0], 
            activation = activation,
            )
        self.dense_layer1 = tf.keras.layers.Dense(
            nodes[1], 
            activation = activation,
            )
        self.nn_dropout = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(
            nodes[2],
            activation = 'sigmoid',
            ) 


    def call(self, input_features):
        activation = self.flatten_layer(input_features)
        activation = self.dense_layer(activation)
        activation = self.nn_dropout(activation)
        activation = self.dense_layer1(activation)
        activation = self.nn_dropout(activation)
        return self.output_layer(activation)
    
