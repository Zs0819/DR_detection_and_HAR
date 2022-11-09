import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from input_pipeline.create_dataset import create_dataset
import gin


@gin.configurable
def rnn_model(num_rnn_neurons=128, num_rnn_layers=1,  dropout_rate=0.5, num_dense_neurons=10, num_dense_layers=1, rnn_type='lstm'):

    WindowSize, _ = create_dataset(return_mode='params')
    # WindowSize = 250
    model = keras.Sequential()

    return model