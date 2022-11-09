import gin
import tensorflow as tf
import logging
from utils.utils import plot_to_image, image_grid
from evaluation.metrics import BalancedSparseCategoricalAccuracy


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, counts, total_steps, log_interval, ckpt_interval, lr):

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.lr = lr
        # self.classes = [0, 1, 2, 3, 4]
        self.counts = counts
        self.num_classes = len(counts)
        self.inverse_weight = [0] * self.num_classes
        self.total_length_per_resample = sum(counts)
        for i in range(len(self.inverse_weight)):
            self.inverse_weight[i] = self.total_length_per_resample / self.counts[i]
        # self.normal_inverse_weight = [float(i)/sum(self.inverse_weight) for i in self.inverse_weight]
        self.train_counts = [0] * self.num_classes
        self.test_counts = [0] * self.num_classes

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                         reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = BalancedSparseCategoricalAccuracy(name='test_accuracy')

        # Summary Writer
        self.train_summary_writer = tf.summary.create_file_writer(self.run_paths["path_summary_train"])
        self.test_summary_writer = tf.summary.create_file_writer(self.run_paths["path_summary_eval"])
        self.image_summary_writer = tf.summary.create_file_writer(self.run_paths["path_summary_image"])

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.checkpoint_path = self.run_paths["path_ckpts_train"]
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.checkpoint_path, max_to_keep=5)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
            factor = tf.gather(self.inverse_weight, labels)
            loss = tf.math.multiply(factor, loss)
            loss = tf.math.reduce_mean(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy.update_state(labels, predictions)
        # self.test_accuracy.result().numpy()

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            if step <= 20:
                print('Training phase')
                print('step {}: label is {}'.format(int(step), labels.numpy()))
                for label in labels:
                    self.train_counts[int(label)] += 1
                    if (self.train_counts[0] + self.train_counts[1]) % self.total_length_per_resample == 0:
                        print(self.train_counts[0], "and", self.train_counts[1])

            # images = tf.clip_by_value(images, clip_value_min=0, clip_value_max=1)
            # figure = image_grid(images, labels, self.classes)
            # with self.image_summary_writer.as_default():
            #     tf.summary.image("visualize images", plot_to_image(figure), step=0)
            self.train_step(images, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                for test_images, test_labels in self.ds_val:
                    print('Validation phase')
                    for label in test_labels:
                        self.test_counts[int(label)] += 1
                        if (self.test_counts[0] + self.test_counts[1]) % 100 == 0:
                            print(self.test_counts[0], "and", self.test_counts[1])
                    self.test_step(test_images, test_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                # Write summary to tensorboard
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)

                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.test_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                saved_path = self.ckpt_manager.save()
                print("save checkpoint for step {} at {}".format(int(step), saved_path))

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint

                saved_path = self.ckpt_manager.save()
                print("save checkpoint for final step {} at {}".format(int(step), saved_path))
                return self.test_accuracy.result().numpy()
