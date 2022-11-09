import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

from input_pipeline.preprocessing import preprocess, augment, fiveToTwo
from input_pipeline.create_smaller_tfrecord import train_record_file, test_record_file, dataType


@gin.configurable
def load(name, data_dir, num_classes):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # load tfrecord file
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
            image = tf.io.decode_jpeg(record['image_raw'], channels=3)
            # image = tf.io.decode_raw(record['image_raw'], out_type=dataType)
            label = record['label']
            height = record['height']
            width = record['width']
            channel = record['channel']
            image = tf.reshape(image, (height, width, channel))

            return image, label

        # create the dataset object from tfrecord file
        unbalanced_ds_train = tf.data.TFRecordDataset(train_record_file)
        unbalanced_ds_train = unbalanced_ds_train.map(parse_record).map(decode_record)

        unbalanced_ds_test = tf.data.TFRecordDataset(test_record_file)
        unbalanced_ds_test = unbalanced_ds_test.map(parse_record).map(decode_record)

        # split train dataset into train and validation dataset
        ds_val = unbalanced_ds_train.take(100)
        ds_train = unbalanced_ds_train.skip(100)
        ds_test = unbalanced_ds_test

        if num_classes == 2:
            ds_train = ds_train.map(fiveToTwo)
            ds_val = ds_val.map(fiveToTwo)
            ds_test = ds_test.map(fiveToTwo)

        # balanced train dataset
        def class_func(image, label):
            return label

        target_dist = [1/num_classes] * num_classes
        resampler = tf.data.experimental.rejection_resample(class_func, target_dist, seed=1234)
        ds_train = ds_train.apply(resampler)
        ds_train = ds_train.map(lambda extra_label, image_and_label: image_and_label)
        # ds_val = ds_train.take(100)
        # ds_train = ds_train.skip(100)

        counts = [0] * num_classes
        for _, label in ds_train.take(10000):
            counts[int(label)] += 1

        counts = [0] * num_classes
        for _, label in ds_train.take(10000):
            counts[int(label)] += 1

        # no need to balance for test dataset
        # ds_test = unbalanced_ds_test.apply(resampler)
        # ds_test = ds_test.map(lambda extra_label, image_and_label: image_and_label)
        # ds_test = unbalanced_ds_test

        ds_info = 'IDRID'

        ds_train, ds_val, ds_test, ds_info = prepare(ds_train, ds_val, ds_test, ds_info)
        return ds_train, ds_val, ds_test, ds_info, counts

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.shuffle(100)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
