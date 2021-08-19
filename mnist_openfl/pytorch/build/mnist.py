"""
https://www.tensorflow.org/tutorials/quickstart/beginner
Disable GPUs - not all nodes used for testing have GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = ''
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import sys
import yaml
import json
import os
import logging
import logging.config
import argparse
from enum import Enum
from typing import List
import wget
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def test(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            test_loss += ce_loss(output, target).item() * data.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(data_loader.sampler)
    print('\nLoss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    accuracy = correct / len(data_loader.dataset)
    return {"loss": test_loss, "accuracy": accuracy.item()}


def train_loop(n_epochs, model, optimizer, train_loader, log_interval, save_path):
    model.train()
    train_loss = 0.0
    ce_loss = nn.CrossEntropyLoss()
    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = ce_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)  # update training loss
            if batch_idx % log_interval == 0 or batch_idx + 1 == len(train_loader):
                message = 'Train Epoch: {}/{} [{}/{} ({:.0f}%)]    Loss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())
                sys.stdout.write("\r" + message)
                sys.stdout.flush()
        print("")
        

def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def bar_custom(current, total, width=80):
    """Custom progress bar for displaying download progress"""
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


class Task(str, Enum):
    DownloadData = 'download'
    Train = 'train'
    Evaluate = 'evaluate'
    
    
def download(task_args: List[str]) -> None:
    """ Task: download.
    Input parameters:
        --data_dir
    """
    logger.info(f"Starting '{Task.DownloadData}' task")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '--data-dir', type=str,
                        default=None, help="Path to a dataset file.")
    args = parser.parse_args(args=task_args)

    if args.data_dir is None:
        raise ValueError(
            "Data directory is not specified (did you use --data-dir=PATH?)")
    os.makedirs(args.data_dir, exist_ok=True)

    if not args.data_dir.startswith("/"):
        logger.warning("Data directory seems to be a relative path.")

    data_file = os.path.join(args.data_dir, 'mnist.npz')
    if os.path.exists(data_file):
        logger.info(
            "MNIST data has already been download (file exists: %s)", data_file)
        return

    data_file = wget.download(
        'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', data_file, bar=bar_custom)

    if not os.path.isfile(data_file):
        raise ValueError(
            f"MNIST dataset has not been downloaded - dataset file does not exist: {data_file}")
    else:
        logger.info("MNIST dataset has been downloaded.")
    logger.info("The '%s' task has been completed.", Task.DownloadData)


def train(task_args: List[str]) -> None:
    """ Task: train.
    Input parameters:
        --data_dir, --log_dir, --model_dir, --parameters_file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '--data-dir', type=str,
                        default=None, help="Dataset path.")
    parser.add_argument('--model_in', '--model-in', type=str,
                        default=None, help="Model output directory.")
    parser.add_argument('--model_dir', '--model-dir', type=str,
                        default=None, help="Model output directory.")
    parser.add_argument('--parameters_file', '--parameters-file', type=str, default=None,
                        help="Parameters default values.")
    parser.add_argument('--metrics', '--metrics', type=str, default=None,
                        help="Parameters default values.")

    args = parser.parse_args(args=task_args)

    with open(args.parameters_file, 'r') as stream:
        parameters = yaml.load(stream, Loader=yaml.FullLoader)
    logger.info("Parameters have been read (%s).", args.parameters_file)

    dataset_file = os.path.join(args.data_dir, 'mnist.npz')
    with np.load(dataset_file, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
    x_train = x_train / 255.0

    tensor_x = torch.Tensor(x_train)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train).type(torch.LongTensor)
    train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = DataLoader(train_dataset, parameters.get('batch_size', 32))

    logger.info("Dataset has been loaded (%s).", dataset_file)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    if args.model_in != '' and len(os.listdir(args.model_in)) != 0:
        # Load from checkpoint;
        model_path = os.path.join(args.model_in, 'model.pth')
        model.load_state_dict(torch.load(model_path))
        optimizer_path = os.path.join(args.model_in, 'optimizer.pth')
        optimizer.load_state_dict(torch.load(optimizer_path))

    logger.info("Model has been built.")

    n_epochs = parameters.get('epochs', 5)
    train_loop(n_epochs, model, optimizer,
                         train_loader, 50, args.model_dir)

    logger.info("Model has been trained.")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(
        args.model_dir, 'optimizer.pth'))
    logger.info("Model has been saved.")

    metrics = test(model, train_loader)

    with open(args.metrics, 'w') as f:
        data_json = {'loss': str(metrics["loss"]),
                    'accuracy': str(metrics["accuracy"])}
        json.dump(data_json, f)


def evaluate(task_args: List[str]) -> None:
    """ Task: train.
    Input parameters:
        --data_dir, --log_dir, --model_dir, --parameters_file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '--data-dir', type=str,
                        default=None, help="Dataset path.")
    parser.add_argument('--model_in', '--model-in', type=str,
                        default=None, help="Model output directory.")
    parser.add_argument('--parameters_file', '--parameters-file', type=str, default=None,
                        help="Parameters default values.")
    parser.add_argument('--metrics', '--metrics', type=str, default=None,
                        help="Parameters default values.")
    args = parser.parse_args(args=task_args)

    with open(args.parameters_file, 'r') as stream:
        parameters = yaml.load(stream, Loader=yaml.FullLoader)
    logger.info("Parameters have been read (%s).", args.parameters_file)

    dataset_file = os.path.join(args.data_dir, 'mnist.npz')
    with np.load(dataset_file, allow_pickle=True) as f:
        x_test, y_test = f['x_test'], f['y_test']
    x_test = x_test / 255.0

    tensor_x = torch.Tensor(x_test)  # transform to torch tensor
    tensor_y = torch.Tensor(y_test).type(torch.LongTensor)
    eval_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    eval_loader = DataLoader(eval_dataset, parameters.get('batch_size', 32))

    logger.info("Dataset has been loaded (%s).", dataset_file)

    model = Net()
    model_path = os.path.join(args.model_in, 'model.pth')
    model.load_state_dict(torch.load(model_path))

    metrics = test(model, eval_loader)

    with open(args.metrics, 'w') as f:
        data_json = {'loss': str(metrics["loss"]),
                    'accuracy': str(metrics["accuracy"])}
        json.dump(data_json, f)

    logger.info("Model has been evaluated.")


def main():
    """
    mnist.py task task_specific_parameters...
    """
    # noinspection PyBroadException
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('mlcube_task', type=str,
                            help="Task for this MLCube.")
        parser.add_argument('--log_dir', '--log-dir', type=str,
                            required=True, help="Logging directory.")
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

        if mlcube_args.mlcube_task == Task.DownloadData:
            download(task_args)
        elif mlcube_args.mlcube_task == Task.Train:
            train(task_args)
        elif mlcube_args.mlcube_task == Task.Evaluate:
            evaluate(task_args)
        else:
            raise ValueError(f"Unknown task: {task_args}")
    except Exception as err:
        logger.exception(err)


if __name__ == '__main__':
    main()
