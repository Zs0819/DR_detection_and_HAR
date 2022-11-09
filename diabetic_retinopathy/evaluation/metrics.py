import numpy as np
import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight("confusion_matrix", shape=(num_classes, num_classes), initializer="zeros")

    def reset_state(self):
        for state in self.variables:
            state.assign(tf.zeros(shape=state.shape))

    def update_state(self, y, y_pred):
        # convert from possibility to boolean
        # y_pred = tf.math.argmax(y_pred, axis=1)
        confusion_matrix = tf.math.confusion_matrix(y, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        # confusion_matrix = tf.transpose(confusion_matrix)
        self.confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        return self.confusion_matrix

    def calculateConfusionMatrix(self, y, y_pred):
        return tf.math.confusion_matrix(y, y_pred, dtype=tf.float32, num_classes=self.num_classes)

    def processConfusionMatrix(self):
        confusion_matrix = self.confusion_matrix
        diag_part = tf.linalg.diag_part(confusion_matrix)
        precision = diag_part / (tf.reduce_sum(confusion_matrix, 0) + tf.constant(1e-10))
        precision = precision.numpy()[1]
        sensitivity_specificity = diag_part / (tf.reduce_sum(confusion_matrix, 1) + tf.constant(1e-10))
        sensitivity = sensitivity_specificity.numpy()[1]
        specificity = sensitivity_specificity.numpy()[0]
        f1 = 2 * precision * sensitivity / (precision + sensitivity + tf.constant(1e-10))

        return precision, sensitivity, specificity, f1


class BalancedSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)
