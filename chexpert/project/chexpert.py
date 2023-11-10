import os
import yaml
import sys
import argparse
import logging
import logging.config
import json
import time
from tqdm import tqdm
from enum import Enum
from typing import List
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

logger = logging.getLogger(__name__)


class Task(str, Enum):
    DownloadData = "download_data"
    DownloadCkpt = "download_ckpt"
    Infer = "infer"


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def infer(task_args: List[str]) -> None:
    """ Task: infer

    Input parameters:
        --data_dir, --ckpt_dir, --out_dir
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "--data-dir", type=str, default=None, help="Dataset path."
    )
    parser.add_argument(
        "--model_dir", "--model-dir", type=str, default=None, help="Model location."
    )
    parser.add_argument(
        "--out_dir", "--out-dir", type=str, default=None, help="Model output directory."
    )

    args = parser.parse_args(args=task_args)
    run(args)


def get_pred(output, cfg):
    if cfg.criterion == "BCE" or cfg.criterion == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    elif cfg.criterion == "CE":
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output)
        pred = prob[:, 1].cpu().detach().numpy()
    else:
        raise Exception("Unknown criterion : {}".format(cfg.criterion))

    return pred


def test_epoch(cfg, model, device, dataloader, out_csv_path):
    torch.set_grad_enabled(False)
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    test_header = [
        "Path",
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]

    with open(out_csv_path, "w") as f:
        f.write(",".join(test_header) + "\n")
        for step in tqdm(range(steps)):
            image, path = next(dataiter)
            image = image.to(device)
            output, __ = model(image)
            batch_size = len(path)
            pred = np.zeros((num_tasks, batch_size))

            for i in range(num_tasks):
                pred[i] = get_pred(output[i], cfg)

            for i in range(batch_size):
                batch = ",".join(map(lambda x: "{}".format(x), pred[:, i]))
                result = path[i] + "," + batch
                f.write(result + "\n")
                logging.info(
                    "{}, Image : {}, Prob : {}".format(
                        time.strftime("%Y-%m-%d %H:%M:%S"), path[i], batch
                    )
                )


def run(args):
    ckpt_path = os.path.join(args.model_dir, "model.pth")
    config_path = os.path.join(args.model_dir, "config.json")
    print(config_path)
    with open(config_path) as f:
        cfg = edict(json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = Classifier(cfg).to(device).eval()
    model.load_state_dict(ckpt)

    out_csv_path = os.path.join(args.out_dir, "inferences.csv")
    in_csv_path = os.path.join(args.data_dir, "valid.csv")

    dataloader_test = DataLoader(
        ImageDataset(in_csv_path, cfg, args.data_dir, mode="test"),
        batch_size=cfg.dev_batch_size,
        drop_last=False,
        shuffle=False,
    )

    test_epoch(cfg, model, device, dataloader_test, out_csv_path)


def main():
    """
    chexpert.py task task_specific_parameters...
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir", "--log-dir", type=str, required=True, help="Logging directory."
    )
    mlcube_args, task_args = parser.parse_known_args()

    os.makedirs(mlcube_args.log_dir, exist_ok=True)
    logger_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": os.path.join(
                    mlcube_args.log_dir, f"mlcube_chexpert_infer.log"
                ),
            }
        },
        "loggers": {
            "": {"level": "INFO", "handlers": ["file_handler"]},
            "__main__": {"level": "NOTSET", "propagate": "yes"},
            "tensorflow": {"level": "NOTSET", "propagate": "yes"},
        },
    }
    logging.config.dictConfig(logger_config)
    infer(task_args)


if __name__ == "__main__":
    main()
