import gin
import logging
import tensorflow as tf
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.vgg_like import vgg_like
from models.resnet_like import resnet_like
from models.transfer_models import densenet_transfer

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

FLAGS = flags.FLAGS
flags.DEFINE_string('mode', default='train', help="Choose from ['train', 'evaluate']")
flags.DEFINE_string('model_name', default='vgg', help="Choose from ['vgg', 'resnet', 'transfer']")
flags.DEFINE_integer('num_classes', default=2, help="Choose from [2, 5]")


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    if FLAGS.num_classes == 2:
        if FLAGS.model_name == 'vgg':
            gin.parse_config_files_and_bindings(['configs/config_2classes.gin'], [])
            checkpoint_path = './checkpoint/2classes/vgg'
            model = vgg_like(input_shape=(512, 512, 3))

        elif FLAGS.model_name == 'resnet':
            gin.parse_config_files_and_bindings(['configs/config_2classes.gin'], [])
            checkpoint_path = './checkpoint/2classes/resnet'
            model = resnet_like(input_shape=(512, 512, 3), training=True)

        else:
            gin.parse_config_files_and_bindings(['configs/config_2classes.gin'], [])
            checkpoint_path = './checkpoint/2classes/transfer'
            model = densenet_transfer(input_shape=(512, 512, 3))
    else:
        gin.parse_config_files_and_bindings(['configs/config_5classes.gin'], [])
        checkpoint_path = './checkpoint/5classes'
        model = densenet_transfer(input_shape=(512, 512, 3))

    # if FLAGS.model_name == 'vgg':
    #     if FLAGS.num_classes == 2:
    #         gin.parse_config_files_and_bindings(['configs/config_2classes.gin'], [])
    #         checkpoint_path = './checkpoint/2classes/vgg'
    #     else:
    #         pass
    # elif FLAGS.model_name == 'resnet':
    #     if FLAGS.num_classes == 2:
    #         gin.parse_config_files_and_bindings(['configs/config_2classes.gin'], [])
    #         checkpoint_path = './checkpoint/2classes/resnet'
    #     else:
    #         pass
    # else:
    #     if FLAGS.num_classes == 2:
    #         gin.parse_config_files_and_bindings(['configs/config_2classes.gin'], [])
    #         checkpoint_path = './checkpoint/2classes/transfer'
    #     else:
    #         pass

    utils_params.save_config(run_paths['path_gin'], gin.operative_config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info, counts = datasets.load()

    # # model
    # if FLAGS.model_name == 'vgg':
    #     model = vgg_like(input_shape=(512, 512, 3))
    # elif FLAGS.model_name == 'resnet':
    #     model = resnet_like(input_shape=(512, 512, 3), training=True)
    # else:
    #     model = densenet_transfer(input_shape=(512, 512, 3))

    if FLAGS.mode == 'train':
        if FLAGS.model_name == 'transfer' or FLAGS.num_classes == 5:
            trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, counts, lr=1e-6)
        else:
            trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, counts, lr=1e-3)
        for _ in trainer.train():
            continue
    else:
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
        evaluate(model, checkpoint, ds_test, checkpoint_path, run_paths)


if __name__ == "__main__":
    app.run(main)
