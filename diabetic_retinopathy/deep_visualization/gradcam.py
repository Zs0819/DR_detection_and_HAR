import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import ResNet50


image_path = 'D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/images/train/IDRiD_001.jpg'
image = np.array(load_img(image_path, target_size=(224, 224, 3)))
# plt.imshow(image)
# plt.show()

model = ResNet50()
# print(model.summary())

# Grad-CAM
# create a model that goes up to the last convolution layer
last_conv_layer = model.get_layer('conv5_block3_out')
last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

# create a model that takes the output of the model above
classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in ['avg_pool', 'predictions']:
    x = model.get_layer(layer_name)(x)
classifier_model = tf.keras.Model(classifier_input, x)

with tf.GradientTape() as tape:
    inputs = image[np.newaxis, ...]
    # print(inputs.shape)
    last_conv_layer_output = last_conv_layer_model(inputs)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    # print(top_pred_index)
    top_class_channel = preds[:, top_pred_index]
    # print(top_class_channel)

grads = tape.gradient(top_class_channel, last_conv_layer_output)
# print(grads.shape)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
# print(last_conv_layer_output.shape)
last_conv_layer_output = last_conv_layer_output.numpy()[0]
# print(last_conv_layer_output)
pooled_grads = pooled_grads.numpy()
# print(pooled_grads.shape[-1])
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

# average over all the filters to get a single 2D array
gradcam = np.mean(last_conv_layer_output, axis=-1)
# print(gradcam)
# clip the values (equivalent to ReLU) and normalize
gradcam = np.clip(gradcam, 0, np.max(gradcam)) / (np.max(gradcam) + 1e-9)
# print(gradcam)
gradcam = cv2.resize(gradcam, (224, 224))
plt.imshow(image)
plt.imshow(gradcam, alpha=0.5)
plt.show()
