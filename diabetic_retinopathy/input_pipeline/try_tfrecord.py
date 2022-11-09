import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train / 255.
x_test = x_test / 255.

# Track the data type
dataType = x_train.dtype
print(f"Data type: {dataType}")

labelType = y_train.dtype
print(f'Data type: {labelType}')


# convert values to compatible tf.Example types
def _bytes_feature(value):
    # return a bytes_list form a string / byte
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    # return a float_list from a float / double
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    # return an int64_list from a bool / enum / int / unit
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# create the features dictionary
def image_example(image, label, dimension):
    feature = {
        'dimension': _int64_feature(dimension),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image.tobytes()),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

record_file = 'mnistTrain.tfrecords'
n_samples = x_train.shape[0]
dimension = x_train.shape[1]
with tf.io.TFRecordWriter(record_file) as writer:
    for i in range(n_samples):
        image = x_train[i]
        label = y_train[i]
        tf_example = image_example(image, label, dimension)
        writer.write(tf_example.SerializeToString())


# decoding function
def parse_record(record):
    name_to_features = {
        'dimension': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    return tf.io.parse_single_example(record, name_to_features)


def decode_record(record):
    image = tf.io.decode_raw(
        record['image_raw'], out_type=dataType, little_endian=True, fixed_length=None, name=None
    )
    label = record['label']
    dimension = record['dimension']
    image = tf.reshape(image, (dimension, dimension))

    return image, label

# create the dataset object from tfrecord file
dataset = tf.data.TFRecordDataset(record_file)
dataset = dataset.map(parse_record).map(decode_record)

# visualization
for image, label in dataset:
    plt.imshow(image)
    plt.show()
    break

