import tensorflow as tf
import gin
from tensorflow.keras import regularizers


def regularized_conv2D(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, kernel_regularizer=regularizers.l2(5e-5),
                                  use_bias=False, kernel_initializer='he_normal')


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = regularized_conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=stride,
                                        padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = regularized_conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(regularized_conv2D(filters=filter_num,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(inputs, filter_num, blocks, stride=1, training=None):
    # res_block = tf.keras.Sequential()
    x = BasicBlock(filter_num, stride=stride)(inputs, training=training)

    for i in range(1, blocks):
        x = BasicBlock(filter_num, stride=1)(x, training=training)

    return x


@gin.configurable
def resnet_like(input_shape, n_classes, base_filter, layers_params, dropout_rate, training=None):
    conv1 = regularized_conv2D(filters=base_filter,
                               kernel_size=(7, 7),
                               strides=2,
                               padding="same")
    bn1 = tf.keras.layers.BatchNormalization()
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding="same")

    # layer1 = make_basic_block_layer(filter_num=64,
    #                                      blocks=layers_params[0])
    # layer2 = make_basic_block_layer(filter_num=128,
    #                                      blocks=layers_params[1],
    #                                      stride=2)
    # layer3 = make_basic_block_layer(filter_num=256,
    #                                      blocks=layers_params[2],
    #                                      stride=2)
    # layer4 = make_basic_block_layer(filter_num=512,
    #                                      blocks=layers_params[3],
    #                                      stride=2)

    avgpool = tf.keras.layers.GlobalAveragePooling2D(name='avg')
    fc = tf.keras.layers.Dense(units=n_classes, name='dense1')
    inputs = tf.keras.Input(input_shape)
    x = conv1(inputs)
    x = bn1(x, training=training)
    x = tf.nn.relu(x)
    x = pool1(x)
    for i, layers in enumerate(layers_params):
        if i == 0:
            x = make_basic_block_layer(inputs=x, filter_num=base_filter, blocks=layers, training=training)
            continue
        x = make_basic_block_layer(inputs=x, filter_num=base_filter * 2 ** i, blocks=layers, stride=2,
                                   training=training)
    # x = layer1(x, training=training)
    # x = layer2(x, training=training)
    # x = layer3(x, training=training)
    # x = layer4(x, training=training)
    x = avgpool(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout1')(x)
    outputs = fc(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet_like')


if __name__ == "__main__":
    model = resnet_like(input_shape=(512, 512, 3), n_classes=2, base_filter=16, layers_params=[2, 2, 2],
                        dropout_rate=0.5, training=True)
    model.summary()
