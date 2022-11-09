import gin
import sklearn.metrics
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import os
import sklearn
import seaborn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from evaluation.metrics import ConfusionMatrix


@gin.configurable
def evaluate(model, checkpoint, ds_test, checkpoint_path, run_paths, num_classes):
    confusion_matrix_test = ConfusionMatrix(num_classes=num_classes)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # compute loss and accuracy and confusion matrix
    for image, label in ds_test:
        loss, accuracy = model.evaluate(image, label, verbose=0)
        label_pred = model(image, training=False)
        preds = tf.keras.activations.softmax(label_pred)
        label_pred = tf.math.argmax(label_pred, -1)
        # print(label_pred.shape)
        # print(label.shape)
        confusion_matrix_test.update_state(label, label_pred)

    print('Accuracy on test dataset: %5.3f' % (accuracy_score(label, label_pred)))
    print(classification_report(label, label_pred))

    # template = 'Test Loss: {}, Test Accuracy: {}'
    # logging.info(template.format(loss, accuracy * 100))

    template = 'Confusion Matrix: \n{}'
    logging.info(template.format(confusion_matrix_test.result().numpy()))

    # Confusion Matrix Visualization
    plt.figure()
    cm = np.array(confusion_matrix_test.result().numpy().tolist())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    seaborn.heatmap(cm, annot=True, fmt='.1%')
    plt.xlabel('Predict')
    plt.ylabel('True')
    cm_path = os.path.join(run_paths['path_summary_image'], 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.show()
    # ROC Visualization, for binary classification
    if num_classes == 2:
        roc_path = os.path.join(run_paths['path_summary_image'], 'roc.png')
        plot_roc(label.numpy(), preds.numpy()[:, 1], roc_path)

        template = 'precision: {}, sensitivity: {}, specificity: {}, f1: {}'
        precision, sensitivity, specificity, f1 = confusion_matrix_test.processConfusionMatrix()
        logging.info(template.format(precision, sensitivity, specificity, f1))
    return


def plot_roc(labels, preds, plot_path):
    plt.figure()
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, preds)
    area = sklearn.metrics.roc_auc_score(labels, preds)
    plt.plot(100 * fpr, 100 * tpr, label='ROC curve (area = %0.2f)' % area)
    plt.xlabel('False Positive [%]')
    plt.ylabel('True Positive [%]')
    plt.legend(loc='lower right')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

    plt.savefig(plot_path)
    plt.show()
