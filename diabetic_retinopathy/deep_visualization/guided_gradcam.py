import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from deep_visualization.guided_propagation import GuidedBackprop

'''
For our own task:
1. change the model
2. change the input size
3. change the layer name
'''

image_path = 'D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/images/train/IDRiD_001.jpg'
image = np.array(load_img(image_path, target_size=(224, 224, 3)))
plt.imshow(image)
plt.show()

model = ResNet50()
# print(model.summary())

# create a model that goes up to the last convolution layer
last_conv_layer = model.get_layer('conv5_block3_out')
last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

# create a model that takes the output of the model above
classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in ['avg_pool', 'predictions']:
    x = model.get_layer(layer_name)(x)
classifier_model = tf.keras.Model(classifier_input, x)

'''
Guided Grad-CAM:
The Grad-CAM output can be improved further by combining with guided backpropagation,
which zeroes elements in the gradients which act negatively towards the decision.
'''
with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(image[np.newaxis, ...])
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)[0]
last_conv_layer_output = last_conv_layer_output[0]

# guided backpropagation implementation
guided_grads = (
        tf.cast(last_conv_layer_output > 0, 'float32')
        * tf.cast(grads > 0, 'float32')
        * grads
)

pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1))
guided_gradcam = np.ones(last_conv_layer_output.shape[:2], dtype=np.float32)
for i, w in enumerate(pooled_guided_grads):
    guided_gradcam += w * last_conv_layer_output[:, :, i]
guided_gradcam = cv2.resize(guided_gradcam.numpy(), (224, 224))
guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
        guided_gradcam.max() - guided_gradcam.min()
)
plt.imshow(image)
plt.imshow(guided_gradcam, alpha=0.5)
plt.show()

'''
# Guided Grad-CAM (high resolution maps):
This approach reflects the paper's description better by first using the guided backpropagation approach
to produce a high resolution map that is of the same resolution of the input image, which is then masked 
using the Grad-CAM heatmap to focus only on details that led to the prediction outcome.
'''
gb = GuidedBackprop(model, "conv5_block3_out")
saliency_map = gb.guided_backprop(image[np.newaxis, ...]).numpy()
saliency_map = saliency_map * np.repeat(guided_gradcam[..., np.newaxis], 3, axis=2)

saliency_map -= saliency_map.mean()
saliency_map /= (saliency_map.std() + 1e-9)
saliency_map *= 0.25
saliency_map += 0.5
saliency_map = np.clip(saliency_map, 0, 1)
saliency_map *= (2 ** 8) - 1
saliency_map = saliency_map.astype(np.uint8)

plt.imshow(saliency_map)
plt.show()

