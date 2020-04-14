# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:51:41 2020

@author: Ludovica
"""

import tensorflow as tf
import numpy as np
        
def parse_function(filename, label1, label2, CHANNELS = 3, IMG_SIZE = 224):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 224.0
    # Decode it into a dense vector
    return image_normalized, label1, label2


def create_dataset(filenames, labels1, labels2, is_training=True, IMG_SIZE = 224, \
                   CHANNELS = 3, BATCH_SIZE = 56, \
                   AUTOTUNE = tf.data.experimental.AUTOTUNE, SHUFFLE_BUFFER_SIZE = 1024):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    filenames = np.asarray(filenames)
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels1, labels2))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
