import tensorflow as tf
import pickle
from random import shuffle
import cv2
import random
import pandas as pd
import numpy as np
import time
tf.enable_eager_execution()
"""import matplotlib.pyplot as plt
import matplotlib.image as mpimg"""

shuffle_data = True
image_size = 96
mean_image = np.zeros([64, 64, 3])

data = pd.read_csv("train_labels.csv")

addrs_t = data['id'].values.tolist()
for i in range(len(addrs_t)):
    addrs_t[i] = 'train/' + addrs_t[i] + '.jpg'

labels = data['label'].values.tolist()

# shuffle the data
if shuffle_data:
    c = list(zip(addrs_t, labels))
    shuffle(c)
    addrs_t, labels = zip(*c)

train_addrs = addrs_t[0:int(0.8 * len(addrs_t))]
train_labels = labels[0:int(0.8 * len(labels))]
# dictionary with train_set
train_dictionary = dict(zip(train_addrs, train_labels))

val_addrs = addrs_t[int(0.8 * len(addrs_t)):]
val_labels = labels[int(0.8 * len(labels)):]
# dictionary with val_set
val_dictionary = dict(zip(val_addrs, val_labels))


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# function to crop images
def crop_centrt(img):
    image_cropped = 64
    cnt_c = (image_size - image_cropped) // 2
    patched_img = img[cnt_c:cnt_c + image_cropped, cnt_c:cnt_c + image_cropped]
    return patched_img

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):

    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

# Write the raw image files to train_images.tfrecords.
# First, process the two images into tf.Example messages.
# Then, write to a .tfrecords file.
writepath = 'dataset/train_images.tfrecords'
with tf.python_io.TFRecordWriter(writepath) as writer:
    for filename, label in train_dictionary.items():
        img = cv2.imread(filename)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        img = crop_centrt(img)
        mean_image += img
        img_str = cv2.imencode('.jpg', img)[1].tostring()
        tf_example = image_example(img_str, label)
        writer.write(tf_example.SerializeToString())


# calculate and write mean
mean_image = mean_image/len(train_addrs)
writepath = 'dataset/mean_v'
with open(writepath, 'wb') as fp:
    pickle.dump(mean_image, fp)


# Write the raw image files to val_images.tfrecords.
# First, process the two images into tf.Example messages.
# Then, write to a .tfrecords file.
writepath = 'dataset/val_images.tfrecords'
with tf.python_io.TFRecordWriter(writepath) as writer:
    for filename, label in val_dictionary.items():
        img = cv2.imread(filename)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        img = crop_centrt(img)
        img_str = cv2.imencode('.jpg', img)[1].tostring()
        tf_example = image_example(img_str, label)
        writer.write(tf_example.SerializeToString())
