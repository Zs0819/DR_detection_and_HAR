import gin
import logging
import tensorflow as tf
from absl import app, flags
from input_pipeline.dataset import load
from train import Trainer
from evaluation.eval import evaluate

from utils import utils_params, utils_misc
from models.rnn_model import rnn_model


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

FLAGS = flags.FLAGS
flags.DEFINE_string('mode', default='train', help="Choose from ['train', 'evaluate']")
flags.DEFINE_string('model_name', default='lstm', help="Choose from ['lstm', 'gru']")


def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    if FLAGS.model_name == 'gru':
        gin.parse_config_files_and_bindings(['configs/config_gru.gin'], [])
        checkpoint_path = './checkpoint/gru'
    else:
        gin.parse_config_files_and_bindings(['configs/config_lstm.gin'], [])
        checkpoint_path = './checkpoint/lstm'

    utils_params.save_config(run_paths['path_gin'], gin.operative_config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = rnn_model()
    model.summary()

    if FLAGS.mode == 'train':
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
        evaluate(model, checkpoint, ds_test, checkpoint_path, run_paths)


if __name__ == "__main__":
    app.run(main)
