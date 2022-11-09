import logging
import gin

import ray
from ray import tune

from input_pipeline.dataset import load
from models.rnn_model import rnn_model
from train import Trainer
from utils import utils_params, utils_misc


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    # change path to absolute path of config file
    # path on dl-lab server
    # gin.parse_config_files_and_bindings(['/misc/home/RUS_CIP/st169623/dl-lab-21w-team10/human_activity_recognition/configs/config.gin'], bindings)
    # path on iss-student server
    gin.parse_config_files_and_bindings(['/no_backups/s1397/dllab/human_activity_recognition/configs/config.gin'], bindings)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = rnn_model()

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


# ray.init(num_cpus=10, num_gpus=1)
analysis = tune.run(
    train_func, num_samples=30, resources_per_trial={"cpu": 10, "gpu": 1},
    config={
        "rnn_model.num_rnn_neurons": tune.choice([64, 128, 256, 512]),
        "rnn_model.num_rnn_layers": tune.choice([1, 2, 3]),
        "rnn_model.num_dense_neurons": tune.choice([64, 128, 256, 512]),
        "rnn_model.num_dense_layers": tune.choice([1, 2, 3]),
        "rnn_model.dropout_rate": tune.uniform(0, 0.9),
        # "rnn_model.rnn_type": tune.choice(['lstm', 'gru']),
        # "create_dataset.window_size": tune.choice([200, 250, 300]),
        # "create_dataset.shift_window_ratio": tune.choice([0.3, 0.5, 0.7]),
        # "prepare.batch_size": tune.choice([32, 64]),
        # "Trainer.total_steps": tune.grid_search([1e4]),
    })

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
