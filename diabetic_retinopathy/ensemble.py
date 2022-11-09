import tensorflow as tf
from models.vgg_like import vgg_like
from models.resnet_like import resnet_like
from models.transfer_models import densenet_transfer
from evaluation.metrics import ConfusionMatrix
# from models.architectures import vgg_like
from input_pipeline import datasets
from sklearn.metrics import accuracy_score
import numpy as np
import gin
from utils import utils_params, utils_misc
import logging
import seaborn
import matplotlib.pyplot as plt
import os
from absl import app, flags
# from evaluation.eval import plot_roc

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_classes', default=2, help="Choose from [2, 5]")


def ensemble_learning(argv):
    def ensemble_averaging(ds_test, models, num_classes):
        """ Combine models by averaging the predictions
        Parameters:
            ds_test (dataset): tensorflow dataset object
            models (tuple: 3 keras.model): the models models that we use for ensemble
            num_models (int): number of models that we use for ensemble
        Returns:
            accuracy_score(float): accuracy of this ensemble model
            test_cm(keras.metric): confusion matrix
            test_labels(np array): true labels of the dataset
            ensembled_predictions(Tensor): predictions of ensemble model
        """

        test_cm = ConfusionMatrix(num_classes)
        for test_images, test_labels in ds_test:
            total_pred_predictions = 0
            for model in models:
                predictions = model(test_images, training=False)
                # print("pred")###
                # tf.print(predictions)######
                # label_pred = np.argmax(predictions, -1)
                # print("predictions")
                # print(predictions)
                total_pred_predictions += predictions
            ensembled_predictions = total_pred_predictions / 3
            # print("average")#####
            # tf.print(ensembled_predictions)######
            # print("ensembled_predictions")
            # print(ensembled_predictions)
            label_pred = np.argmax(ensembled_predictions, -1)
            # print("after argmax")######
            # print(label_pred)######
            test_cm.update_state(test_labels, label_pred)
            accuracy = accuracy_score(test_labels, label_pred)

        return accuracy, test_cm

    def ensemble_voting(ds_test, models, num_classes):
        """ Combine models by voting
        Parameters:
            ds_test (dataset): tensorflow dataset object
            models (tuple: 3 keras.model): the models models that we use for ensemble
            num_models (int): number of models that we use for ensemble
        Returns:
            accuracy_score(float): accuracy of this ensemble model
            test_cm(keras.metric): confusion matrix
            test_labels(np array): true labels of the dataset
            ensembled_predictions(Tensor): predictions of ensemble model
        """
        test_cm = ConfusionMatrix(num_classes=num_classes)
        final_pre = np.zeros((103, 1))
        labels_pre_list = []
        for i in range(3):
            labels_pre_list.append(np.zeros((103, 1)))
        labels = np.zeros((103, 1))
        for test_images, test_labels in ds_test:
            for idx, model in enumerate(models):
                predictions = model(test_images, training=False)
                # label_pred = np.argmax(predictions, -1)
                # print("predictions")
                # print(predictions)
                label_pred = np.argmax(predictions, -1)
                # print(idx, label_pred) ########
                labels_pre_list[idx] = label_pred
            labels = test_labels.numpy()
            # print("labels: ", labels) ########
        labels = tf.convert_to_tensor(labels, dtype=tf.int64)
        ensemble_labels_pre = tf.stack([labels_pre_list[0], labels_pre_list[1], labels_pre_list[2]], axis=1)
        ensemble_labels_pre = tf.squeeze(ensemble_labels_pre)
        # print("after squeeze")######
        # tf.print(ensemble_labels_pre)####
        for idx, each_label_pre in enumerate(ensemble_labels_pre):
            count = np.bincount(each_label_pre, minlength=num_classes)
            if np.max(count) == 1:
                val = ensemble_labels_pre[idx][1]  # if no major votes, follow the prediction of the best performed model vgg_like(idx 1 in models)
                final_pre[idx] = val
                continue
            val = np.argmax(count, -1)
            final_pre[idx] = val

        final_pre = tf.convert_to_tensor(final_pre, tf.int64)
        # print("final")########
        # tf.print(final_pre)######
        accuracy = accuracy_score(labels, final_pre)
        test_cm.update_state(labels, final_pre)

        return accuracy, test_cm

    '''two config files for ensemble learning, 2classes and 5 classes respectively'''
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
    #
    # # gin-config
    # gin.parse_config_files_and_bindings(['/content/drive/MyDrive/diabetic_retinopathy/configs/config.gin'],
    #                                     [])
    # utils_params.save_config(run_paths['path_gin'], gin.config_str())

    if FLAGS.num_classes == 2:
        gin.parse_config_files_and_bindings(['./configs/config_ensemble_2classes.gin'], [])
        checkpoint_path_vgg = './checkpoint/2classes/vgg'
        checkpoint_path_resnet = './checkpoint/2classes/resnet'
        checkpoint_path_transfer = './checkpoint/2classes/transfer'

    else:
        pass

    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    #
    # setup pipeline
    ds_train, ds_val, ds_test, ds_info, counts = datasets.load()

    model1 = resnet_like(input_shape=(512, 512, 3), training=False)
    # model1.build(input_shape=(256, 256, 3))
    checkpoint1 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model1)
    checkpoint1.restore(tf.train.latest_checkpoint(checkpoint_path_resnet)).expect_partial()
    # model1.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
    model1.summary()

    model2 = vgg_like(input_shape=(512, 512, 3))
    # model2.build(input_shape=(None, 256, 256, 3))
    checkpoint2 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model2)
    checkpoint2.restore(tf.train.latest_checkpoint(checkpoint_path_vgg)).expect_partial()
    # model2.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
    model2.summary()

    model3 = densenet_transfer(input_shape=(512, 512, 3))
    checkpoint3 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model3)
    checkpoint3.restore(tf.train.latest_checkpoint(checkpoint_path_transfer)).expect_partial()
    # model3.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
    model3.summary()

    models = [model1, model2, model3]

    acc_average, cm_average = ensemble_averaging(ds_test, models, num_classes=FLAGS.num_classes)
    acc_vote, cm_vote = ensemble_voting(ds_test, models, num_classes=FLAGS.num_classes)

    # Show accuracy for ensemble averaging
    template = 'accuracy for ensemble averaging: {}'
    logging.info(template.format(acc_average))
    print(template.format(acc_average))

    # Show accuracy for ensemble voting
    template = 'accuracy for ensemble voting: {}'
    logging.info(template.format(acc_vote))
    print(template.format(acc_vote))

    # Confusion matrix for ensemble averaging
    template = 'Confusion Matrix for ensemble averaging:\n{}'
    logging.info(template.format(cm_average.result().numpy()))
    print(template.format(cm_average.result().numpy()))

    # Confusion matrix for ensemble voting
    template = 'Confusion Matrix for ensemble voting:\n{}'
    logging.info(template.format(cm_vote.result().numpy()))
    print(template.format(cm_vote.result().numpy()))

    # Confusion Matrix Visualization(ensemble averaging)
    plt.figure()
    cm = np.array(cm_average.result().numpy().tolist())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    seaborn.heatmap(cm, annot=True, fmt='.1%')
    plt.xlabel('Predict')
    plt.ylabel('True')
    cm_path = os.path.join(run_paths['path_summary_image'], 'ensemble_averaging_cm.png')
    plt.savefig(cm_path)
    plt.show()

    # Confusion Matrix Visualization(ensemble voting)
    plt.figure()
    cm = np.array(cm_vote.result().numpy().tolist())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    seaborn.heatmap(cm, annot=True, fmt='.1%')
    plt.xlabel('Predict')
    plt.ylabel('True')
    cm_path = os.path.join(run_paths['path_summary_image'], 'ensemble_voting_cm.png')
    plt.savefig(cm_path)
    plt.show()


if __name__ == "__main__":
    app.run(ensemble_learning)