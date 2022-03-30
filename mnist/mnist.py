import os
import yaml
import uuid
import argparse
import typing as t
import logging
import logging.config
from pathlib import Path
from enum import Enum
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def log(task: str, execution_id: str, event: str, context: t.Optional[t.Dict] = None) -> None:
    logger.info("MLCube name=MNIST, task=%s, execution_id: %s event=%s, context='%s'",
                task, execution_id, event, str(context))


class Task(str, Enum):
    """Tasks implemented in this MLCube"""

    Download = 'download'
    """Download MNIST dataset."""

    Train = 'train'
    """Train a simple neural network model"""


def download(execution_id: str, task_args: t.List[str]) -> None:
    """Download MNIST dataset.
    Args:
        execution_id: Identifier of this task execution.
        task_args: Unparsed task-specific CLI arguments.

    MLCube:
        Input parameters: data_config
        Output parameters: data_dir
    """
    log(Task.Download, execution_id, 'task_started', dict(task_args=task_args))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', '--data-config', type=str, default=None,
                        help="Path to a YAML file with data configuration.")
    parser.add_argument('--data_dir', '--data-dir', type=str, default=None, help="Path to a dataset file.")
    args = parser.parse_args(args=task_args)

    with open(args.data_config, 'r') as stream:
        data_config = yaml.load(stream, Loader=yaml.FullLoader)
    log(Task.Download, execution_id, 'configuration_read', dict(data_config=data_config))

    os.makedirs(args.data_dir, exist_ok=True)
    data_file = tf.keras.utils.get_file(
        fname=Path(args.data_dir) / 'mnist.npz',
        origin=data_config.get('uri', 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'),
        file_hash=data_config.get('hash', '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')
    )
    log(Task.Download, execution_id, 'task_completed', dict(data_file=data_file))


def train(execution_id: str, task_args: t.List[str]) -> None:
    """ Train a classification neural network model.
    Args:
        execution_id: Identifier of this task execution.
        task_args: Unparsed task-specific CLI arguments.

    MLCube:
        Input parameters: data_dir, train_config
        Output parameters: log_dir, model_dir
    """
    log(Task.Train, execution_id, 'task_started', dict(task_args=task_args))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '--data-dir', type=str, default=None, help="Path to training dataset.")
    parser.add_argument('--train_config', '--train-config', type=str, default=None,
                        help="Path to a YAML file with training configuration parameters.")
    parser.add_argument('--model_dir', '--model-dir', type=str, default=None, help="Model output directory.")
    args = parser.parse_args(args=task_args)

    with open(args.train_config, 'r') as stream:
        train_config = yaml.load(stream, Loader=yaml.FullLoader)
    log(Task.Train, execution_id, 'configuration_read', dict(train_config=train_config))

    data_file = Path(args.data_dir) / 'mnist.npz'
    with np.load(str(data_file), allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    x_train, x_test = x_train / 255.0, x_test / 255.0
    log(Task.Train, execution_id, 'data_loaded', dict(data_file=data_file))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=train_config.get('optimizer', 'adam'),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    log(Task.Train, execution_id, 'model_built')

    # Train and evaluate
    model.fit(x_train, y_train, batch_size=train_config.get('batch_size', 32),
              epochs=train_config.get('train_epochs', 5))
    log(Task.Train, execution_id, 'model_trained')

    model.evaluate(x_test, y_test, verbose=2)
    log(Task.Train, execution_id, 'model_evaluated')

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = str(Path(args.model_dir) / 'mnist_model')
    model.save(model_path)
    log(Task.Train, execution_id, 'model_saved', dict(model_path=model_path))


def main():
    """
    mnist.py task task_specific_parameters...
    """
    # noinspection PyBroadException
    parser = argparse.ArgumentParser()
    parser.add_argument('mlcube_task', type=str, help="Task for this MLCube.")
    parser.add_argument('--log_dir', '--log-dir', type=str, required=True, help="Logging directory.")
    mlcube_args, task_args = parser.parse_known_args()

    os.makedirs(mlcube_args.log_dir, exist_ok=True)
    logger_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {"format": "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": os.path.join(mlcube_args.log_dir, f"mlcube_mnist_{mlcube_args.mlcube_task}.log")
            }
        },
        "loggers": {
            "": {"level": "INFO", "handlers": ["file_handler"]},
            "__main__": {"level": "NOTSET", "propagate": "yes"},
            "tensorflow": {"level": "NOTSET", "propagate": "yes"}
        }
    }
    logging.config.dictConfig(logger_config)

    execution_id = str(uuid.uuid4())
    if mlcube_args.mlcube_task == Task.Download:
        download(execution_id, task_args)
    elif mlcube_args.mlcube_task == Task.Train:
        train(execution_id, task_args)
    else:
        raise ValueError(f"Unknown task: {task_args}")
    print(f"MLCube task ({mlcube_args.mlcube_task}) completed. See log file for details.")


if __name__ == '__main__':
    main()
