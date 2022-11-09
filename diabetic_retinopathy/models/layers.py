import gin
import tensorflow as tf
from tensorflow.keras import regularizers


def regularized_conv2D(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, kernel_regularizer=regularizers.l2(0),
                                  use_bias=False, kernel_initializer='he_normal')


@gin.configurable
def vgg_block(inputs, filters, kernel_size=(3, 3)):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    # out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    # out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    # out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = regularized_conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu, name=str(filters)+'_0')(inputs)
    out = regularized_conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu, name=str(filters)+'_1')(out)
    out = tf.keras.layers.MaxPool2D((2, 2), name=str(filters)+'_max')(out)

    return out