import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import augment, preprocess
import numpy as np

# path in dl-lab server
# label_directory = '/misc/home/data/IDRID_dataset/labels/'
# image_directory = '/misc/home/data/IDRID_dataset/images/'

# path in local memory
label_directory = 'D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/labels/'
image_directory = 'D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/images/'

train_df = pd.read_csv(label_directory + 'train.csv')
test_df = pd.read_csv(label_directory + 'test.csv')

train_file_paths = train_df['Image name'].values
train_labels = train_df['Retinopathy grade'].values
ds_train = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))

test_file_paths = test_df['Image name'].values
test_labels = test_df['Retinopathy grade'].values
ds_test = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))


def read_train_image(image_file, label):
    image = tf.io.read_file(image_directory + 'train/' + image_file + '.jpg')
    image = tf.image.decode_jpeg(image)
    return image, label


def read_test_image(image_file, label):
    image = tf.io.read_file(image_directory + 'test/' + image_file + '.jpg')
    image = tf.image.decode_jpeg(image)
    return image, label


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy())
    plt.axis('off')
    plt.show()

ds_train = ds_train.map(read_train_image)
ds_test = ds_test.map(read_test_image)
ds_train = ds_train.map(preprocess).map(augment)

for image, label in ds_train.take(2):
    print(image)
    print(np.max(image))
    show(image, label)
    print(label.numpy())
    print(image.shape)

for image, label in ds_test.take(2):
    show(image, label)
    print(label.numpy())
    print(image.shape)
