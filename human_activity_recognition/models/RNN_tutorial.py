import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# LSTM layer
model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# Output shape (None, None, 64)

# Add a LSTM layer with 128 internal units
model.add(layers.LSTM(128))
# Output shape (None, 128)

# Add a Dense layer with 10 units
model.add(layers.Dense(10))
# Output shape (None, 10)
model.summary()


# RNN layer
model = keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# Output shape (None, None, 64)

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, units)
model.add(layers.GRU(256, return_sequences=True))
# Output shape (None, None, 256)

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, units)
model.add(layers.SimpleRNN(128))
# Output shape (None, 128)

model.add(layers.Dense(10))
# Output shape (None, 10)
model.summary()


# RNN example from github
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# For RNN: SimpleRNN, GRU, LSTM, Bidirectional
model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(None, 28)))
model.add(
    layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, activation='tanh')
    )
)
model.add(
    layers.Bidirectional(
        layers.LSTM(256, activation='tanh')
    )
)
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
