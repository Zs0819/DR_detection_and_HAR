import tensorflow as tf
from input_pipeline.create_dataset import create_dataset

train_record_file = "train.tfrecords"
val_record_file = "val.tfrecords"
test_record_file = "test.tfrecords"


def create_tfrecords(existed_files=False):

    if not existed_files:

        train_x, train_y, window_size, shift_window_size = create_dataset(start_user=1, end_user=21, return_mode='all')
        val_x, val_y, _, _ = create_dataset(start_user=28, end_user=30, return_mode='all')
        test_x, test_y, _, _ = create_dataset(start_user=22, end_user=27, return_mode='all')

        def _bytes_feature(value):
            # return a bytes_list form a string / byte
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


        def data_example(data, label):
            feature = {
                "data": _bytes_feature(data),
                "label": _int64_feature(int(label))
            }
            return tf.train.Example(features=tf.train.Features(feature=feature))



        with tf.io.TFRecordWriter(train_record_file) as writer:
            for i in range(len(train_x.index)):
                data = train_x.iloc[i].values
                data = tf.io.serialize_tensor(data)
                # print(data)
                label = train_y.iloc[i].values
                label = tf.convert_to_tensor(label, dtype=tf.int64)
                # print(label)
                tf_example = data_example(data, label)
                writer.write(tf_example.SerializeToString())

        with tf.io.TFRecordWriter(val_record_file) as writer:
            for i in range(len(val_x.index)):
                data = val_x.iloc[i].values
                data = tf.io.serialize_tensor(data)
                # print(data)
                label = val_y.iloc[i].values
                label = tf.convert_to_tensor(label, dtype=tf.int64)
                # print(label)
                tf_example = data_example(data, label)
                writer.write(tf_example.SerializeToString())

        with tf.io.TFRecordWriter(test_record_file) as writer:
            for i in range(len(test_x.index)):
                data = test_x.iloc[i].values
                data = tf.io.serialize_tensor(data)
                # print(data)
                label = test_y.iloc[i].values
                label = tf.convert_to_tensor(label, dtype=tf.int64)
                # print(label)
                tf_example = data_example(data, label)
                writer.write(tf_example.SerializeToString())

    else:
        window_size, shift_window_size = create_dataset(return_mode='params')

    return window_size, shift_window_size
