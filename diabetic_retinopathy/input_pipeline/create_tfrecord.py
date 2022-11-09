import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# path in dl-lab server
# label_directory = '/misc/home/data/IDRID_dataset/labels/'
# image_directory = '/misc/home/data/IDRID_dataset/images/'

# path in iss-student server
# label_directory = '/no_backups/s1397/dllab/datasets/IDRID_dataset/labels/'
# image_directory = '/no_backups/s1397/dllab/datasets/IDRID_dataset/images/'

# path in local memory
label_directory = 'D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/labels/'
image_directory = 'D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/images/'
train_df = pd.read_csv(label_directory + 'train.csv')
test_df = pd.read_csv(label_directory + 'test.csv')

train_file_paths = train_df['Image name'].values
n_train_samples = len(train_file_paths)
print(f"Number of train samples: {n_train_samples}")
train_labels = train_df['Retinopathy grade'].values
ds_train = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))

test_file_paths = test_df['Image name'].values
n_test_samples = len(test_file_paths)
print(f"Number of test samples: {n_test_samples}")
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

ds_train = ds_train.map(read_train_image)
ds_test = ds_test.map(read_test_image)


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy())
    plt.axis('off')
    plt.show()

for image, label in ds_train.take(1):
    dataType = image.dtype
    print(f"Data type: {dataType}")
    height = image.shape[0]
    print(f"Height: {height}")
    width = image.shape[1]
    print(f"Width: {width}")
    channel = image.shape[2]
    print(f"Channel: {channel}")
    labelType = label.dtype
    print(f"Label type: {labelType}")
    # show(image, label)


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
def image_example(image, label, height, width, channel):
    feature = {
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'channel': _int64_feature(channel),
        'label': _int64_feature(int(label)),
        'image_raw': _bytes_feature(image.numpy().tobytes()),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


train_record_file = 'idrid-train-unbalanced.tfrecords'
test_record_file = 'idrid-test-unbalanced.tfrecords'
with tf.io.TFRecordWriter(train_record_file) as writer:
    for image, label in ds_train:
        # show(image, label)
        height = image.shape[0]
        width = image.shape[1]
        channel = image.shape[2]
        tf_example = image_example(image, label, height, width, channel)
        writer.write(tf_example.SerializeToString())

with tf.io.TFRecordWriter(test_record_file) as writer:
    for image, label in ds_test:
        height = image.shape[0]
        width = image.shape[1]
        channel = image.shape[2]
        tf_example = image_example(image, label, height, width, channel)
        writer.write(tf_example.SerializeToString())


# decoding function
def parse_record(record):
    name_to_features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channel': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    return tf.io.parse_single_example(record, name_to_features)


def decode_record(record):
    image = tf.io.decode_raw(
        record['image_raw'], out_type=dataType, little_endian=True, fixed_length=None, name=None
    )
    label = record['label']
    height = record['height']
    width = record['width']
    channel = record['channel']
    image = tf.reshape(image, (height, width, channel))

    return image, label

# create the dataset object from tfrecord file
dataset_train = tf.data.TFRecordDataset(train_record_file)
dataset_train = dataset_train.map(parse_record).map(decode_record)

ds_val = dataset_train.take(50)
dataset_train = dataset_train.skip(50)

dataset_test = tf.data.TFRecordDataset(test_record_file)
dataset_test = dataset_test.map(parse_record).map(decode_record)


# balanced train dataset
def class_func(image, label):
    return label
num_classes = 5
target_dist = [1 / num_classes] * num_classes
resampler = tf.data.experimental.rejection_resample(class_func, target_dist)
ds_train = dataset_train.apply(resampler)
ds_train = ds_train.map(lambda extra_label, image_and_label: image_and_label)

# no need to balance for test dataset
ds_test = dataset_test
# ds_test = dataset_test.apply(resampler)
# ds_test = ds_test.map(lambda extra_label, image_and_label: image_and_label)

# visualization
counts = [0] * num_classes
print('balanced dataset')
for image, label in ds_train.take(50):
    # print(label.numpy())
    for i in range(num_classes):
        if label == i:
            counts[i] += 1

for i in range(num_classes):
    print(counts[i])

counts = [0] * num_classes
print('unbalanced dataset')
for image, label in dataset_train.take(50):
    # print(label.numpy())
    for i in range(num_classes):
        if label == i:
            counts[i] += 1

for i in range(num_classes):
    print(counts[i])

counts = [0] * num_classes
print('validation dataset')
for image, label in ds_val.take(50):
    # print(label.numpy())
    for i in range(num_classes):
        if label == i:
            counts[i] += 1

for i in range(num_classes):
    print(counts[i])

counts = [0] * num_classes
print('test dataset')
for image, label in ds_test.take(50):
    # print(label.numpy())
    for i in range(num_classes):
        if label == i:
            counts[i] += 1

for i in range(num_classes):
    print(counts[i])

