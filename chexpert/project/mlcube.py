"""MLCube handler file"""
import os
import yaml
import typer
import shutil
import subprocess
from pathlib import Path


app = typer.Typer()

class DownloadModelTask(object):
    """
    Downloads model config and checkpoint files
    Arguments:
    - model_dir [str]: path for storing the model.
    """
    @staticmethod
    def run(model_dir: str) -> None:

        env = os.environ.copy()
        env.update({
            'MODEL_DIR': model_dir,
        })

        process = subprocess.Popen("./download_model.sh", cwd=".", env=env)
        process.wait()

class PreprocessTask(object):
    """
    Task for preprocessing the data
    
    Arguments:
    - data_dir: data location.
    """
    @staticmethod
    def run(data_dir: str) -> None:
        cmd = f"python3.7 preprocess.py --data_dir={data_dir}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()

class InferTask(object):
    """
    Inference task for generating predictions on the CheXpert dataset.

    Arguments:
    - log_dir [str]: logging location.
    - data_dir [str]: data location.
    - model_dir [str]: model location.
    - out_dir [str]: location for storing the predictions.
    """
    @staticmethod
    def run(log_dir: str, data_dir: str, model_dir: str, out_dir) -> None:
        cmd = f"python3.7 chexpert.py --log_dir={log_dir} --data_dir={data_dir} --model_dir={model_dir} --out_dir={out_dir}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()

@app.command("download_model")
def download_model(model_dir: str = typer.Option(..., '--model_dir')):
    DownloadModelTask.run(model_dir)

@app.command("preprocess")
def preprocess(data_dir: str = typer.Option(..., '--data_dir')):
    PreprocessTask.run(data_dir)

@app.command("infer")
def infer(log_dir: str = typer.Option(..., '--log_dir'),
          data_dir: str = typer.Option(..., '--data_dir'),
          model_dir: str = typer.Option(..., '--model_dir'),
          out_dir: str = typer.Option(..., '--out_dir')):
    InferTask.run(log_dir, data_dir, model_dir, out_dir)

if __name__ == '__main__':
    app()