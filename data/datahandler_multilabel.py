# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:51:41 2020

@author: Ludovica
"""

import tensorflow as tf
import numpy as np
import math        
#import tensorflow_addons as tfa


def parse_function(filename, label, CHANNELS = 3, IMG_SIZE = 224,):
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
    return image_normalized, label

def augment(images, labels,
            resize=None, # (width, height) tuple or None
            horizontal_flip=True,
            vertical_flip=False,
            rotate=15, # Maximum rotation angle in degrees
            crop_probability=0.8, # How often we do crops
            crop_min_percent=0.6, # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=4):  #Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)
  
    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        #images = tf.subtract(images, 0.5)
        #images = tf.multiply(images, 2.0)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        height, width = shp[0], shp[1]
        batch_size=8
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
    if horizontal_flip:
        coin = tf.less(tf.random.uniform([batch_size], 0, 1.0), 0.5)
        flip_transform = tf.convert_to_tensor(
              [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
        transforms.append(
        tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if vertical_flip:
        coin = tf.less(tf.random.uniform([batch_size], 0, 1.0), 0.5)
        flip_transform = tf.convert_to_tensor(
              [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
        transforms.append(
              tf.where(coin,
                       tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))
    
    if rotate > 0:
        angle_rad = rotate / 180 * math.pi
        angles = tf.random.uniform([batch_size], -angle_rad, angle_rad)
        transforms.append(
              tfa.image.rotate(
                  angles, height, width))

    if crop_probability > 0:
        crop_pct = tf.random.uniform([batch_size], crop_min_percent,
                                       crop_max_percent)
        left = tf.random.uniform([batch_size], 0, width * (1 - crop_pct))
        top = tf.random.uniform([batch_size], 0, height * (1 - crop_pct))
        crop_transform = tf.stack([
              crop_pct,
              tf.zeros([batch_size]), top,
              tf.zeros([batch_size]), crop_pct, left,
              tf.zeros([batch_size]),
              tf.zeros([batch_size])
        ], 1)

        coin = tf.less(
              tf.random.uniform([batch_size], 0, 1.0), crop_probability)
        transforms.append(
              tf.where(coin, crop_transform,
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if transforms:
        images = tf.image.transform(
              images,
              tfa.image.compose_transforms(*transforms),
              interpolation='BILINEAR') # or 'NEAREST'

    def cshift(values): # Circular shift in batch dimension
        return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

    if mixup > 0:
        mixup = 1.0 * mixup # Convert to float, as tf.distributions.Beta requires floats.
        beta = tf.distributions.Beta(mixup, mixup)
        lam = beta.sample(batch_size)
        ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
        images = ll * images + (1 - ll) * cshift(images)
        labels = lam * labels + (1 - lam) * cshift(labels)

        
    return images, labels

def create_dataset(filenames, labels, is_training=True, IMG_SIZE = 224, \
                   CHANNELS = 3, BATCH_SIZE = 56, \
                   AUTOTUNE = tf.data.experimental.AUTOTUNE, SHUFFLE_BUFFER_SIZE = 1024,):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    filenames = np.asarray(filenames)
    labels = np.asarray(labels)
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    #dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    
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
