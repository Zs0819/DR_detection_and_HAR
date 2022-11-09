import gin
import tensorflow as tf


@gin.configurable
def preprocess(image, label, img_height, img_width):
    # img_height, img_width = 256, 256
    """Dataset preprocessing: Normalizing and resizing"""
    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize_with_pad(image, img_height, img_width)

    return image, label


@gin.configurable
def augment(image, label, img_height, img_width):
    """Data augmentation"""
    # img_height, img_width = 256, 256
    # random jitter
    new_height = int(img_height * 1.15)
    new_width = int(img_width * 1.15)
    image = tf.image.resize(image, size=(new_height, new_width))
    image = tf.image.random_crop(image, size=(img_height, img_width, 3))

    # rotate random angle between (-30, 30)
    # image = tf.keras.preprocessing.image.random_rotation(image, rg=30)
    # image = tfa.image.rotate(image, 0.3)

    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label


def fiveToTwo(image, label):
    if label < 2:
        label = 0
    else:
        label = 1

    return image, label
