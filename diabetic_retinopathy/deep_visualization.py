import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img
from models.resnet_like import resnet_like
from models.vgg_like import vgg_like
from models.transfer_models import densenet_transfer
from guided_propagation import GuidedBackprop
from absl import flags, app


FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', default='vgg', help="Choose from ['vgg', 'resnet']")
flags.DEFINE_integer('num_classes', default=2, help="Choose from [2, 5]")


def deep_visualization(argv):
    if FLAGS.model_name == 'resnet':
        # For resnet model
        targetSize = 512
        channels = 3
        layer_before_classification = 'basic_block_5'
        classification_layers = ['avg', 'dropout1', 'dense1']
        # checkpoint_path = 'D:/Study_in_Germany/Semester_5/dl_lab/result/DR/2classes/resnet/ckpts'
        if FLAGS.num_classes == 2:
            checkpoint_path = './checkpoint/2classes/resnet'
        else:
            pass
        dir = os.path.join(checkpoint_path, 'visualization')
        model = resnet_like(base_filter=16, dropout_rate=0.5, layers_params=[2, 2, 2], n_classes=FLAGS.num_classes, input_shape=(targetSize, targetSize, channels), training=False)
    else:
        # For vgg model
        targetSize = 512
        channels = 3
        layer_before_classification = '128_max'
        classification_layers = ['avg', 'dense1', 'dropout1', 'dense2']
        # checkpoint_path = 'D:/Study_in_Germany/Semester_5/dl_lab/result/DR/2classes/vgg/ckpts'
        if FLAGS.num_classes == 2:
            checkpoint_path = './checkpoint/2classes/vgg'
        else:
            pass
        dir = os.path.join(checkpoint_path, 'visualization')
        model = vgg_like(input_shape=(targetSize, targetSize, channels), n_classes=FLAGS.num_classes, base_filters=16, n_blocks=4, dense_units=64, dropout_rate=0.5)
    # else:
    #     # For transfer model
    #     targetSize = 512
    #     channels = 3
    #     layer_before_classification = 'densenet121'
    #     classification_layers = ['avg', 'dropout1', 'dense1']
    #     if FLAGS.num_classes == 2:
    #         checkpoint_path = './checkpoint/2classes/transfer'
    #     else:
    #         pass
    #     dir = os.path.join(checkpoint_path, 'visualization')
    #     model = densenet_transfer(input_shape=(targetSize, targetSize, channels), num_classes=FLAGS.num_classes, dropout_rate=0.4)

    # Load checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    model.summary()

    # create a model that goes up to the last convolution layer
    last_conv_layer = model.get_layer(layer_before_classification)
    print(model.inputs)
    print(last_conv_layer.output)
    # if FLAGS.model_name == 'transfer':
    #     last_conv_layer_model = tf.keras.Model(inputs=model.inputs, outputs=last_conv_layer.get_output_at(0))
    # else:
    last_conv_layer_model = tf.keras.Model(inputs=model.inputs, outputs=last_conv_layer.output)

    # create a model that takes the output of the model above
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classification_layers:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Images to visualize
    # path on local memory
    # image_paths = ['D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/images/train/IDRiD_001.jpg',
    #                'D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/images/train/IDRiD_002.jpg',
    #                'D:/Study_in_Germany/Semester_5/dl_lab/datasets/IDRID_dataset/images/train/IDRiD_003.jpg']

    # path on the server
    image_paths = ['./datasets/IDRID_dataset/images/train/IDRiD_001.jpg',
                   './datasets/IDRID_dataset/images/train/IDRiD_002.jpg',
                   './datasets/IDRID_dataset/images/train/IDRiD_003.jpg']

    # Images to save
    os.makedirs(dir, exist_ok=True)
    # original_save_path = os.path.join(dir, 'original.png')
    # gradcam_save_path = os.path.join(dir, 'gradcam.png')
    # guided_gradcam_save_path = os.path.join(dir, 'guided_gradcam.png')

    # Processing images
    for idx in range(len(image_paths)):
        image = np.array(load_img(image_paths[idx]))
        image = tf.cast(image, tf.float32) / 255.
        image = tf.image.resize_with_pad(image, targetSize, targetSize)
        plt.imshow(image)
        original_save_path = os.path.join(dir, 'original_' + str(idx+1) + '.png')
        plt.savefig(original_save_path)
        # plt.show()

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
        guided_gradcam = cv2.resize(guided_gradcam.numpy(), (targetSize, targetSize))
        guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
        guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
                guided_gradcam.max() - guided_gradcam.min()
        )
        plt.imshow(image)
        plt.imshow(guided_gradcam, alpha=0.5)
        gradcam_save_path = os.path.join(dir, 'gradcam' + str(idx+1) + '.png')
        plt.savefig(gradcam_save_path)
        # plt.show()

        '''
        # Guided Grad-CAM (high resolution maps):
        This approach reflects the paper's description better by first using the guided backpropagation approach
        to produce a high resolution map that is of the same resolution of the input image, which is then masked 
        using the Grad-CAM heatmap to focus only on details that led to the prediction outcome.
        '''
        gb = GuidedBackprop(model, layer_before_classification)
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
        guided_gradcam_save_path = os.path.join(dir, 'guided_gradcam' + str(idx+1) + '.png')
        plt.savefig(guided_gradcam_save_path)
        # plt.show()

if __name__ == '__main__':
    app.run(deep_visualization)
