from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import DenseNet121
import gin


def top_layers(inputs, dropout_rate, num_classes):
    out = layers.GlobalAveragePooling2D(name='avg')(inputs)
    out = layers.Dropout(dropout_rate, name='dropout1')(out)
    outputs = layers.Dense(num_classes, name='dense1')(out)
    return outputs

@gin.configurable
def densenet_transfer(input_shape, dropout_rate, num_classes):
    """Densenet121 + additional top layers for the classification task"""
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the base model.
    # base_model.trainable = False
    # inputs = Input(shape=input_shape)
    # out = base_model(inputs, training=False)
    # outputs = top_layers(inputs=out, dropout_rate=dropout_rate, num_classes=num_classes)
    # model = Model(inputs, outputs)

    base_model.trainable = True
    inputs = Input(shape=input_shape)
    for layer in base_model.layers[:397]:
        layer.trainable = False
    out = base_model(inputs, training=False)
    outputs = top_layers(inputs=out, dropout_rate=dropout_rate, num_classes=num_classes)
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = densenet_transfer(input_shape=(512, 512, 3), dropout_rate=0.4, num_classes=2)
    model.summary()