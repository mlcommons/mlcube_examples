"""MLCube handler file"""
import os
import yaml
import typer
import shutil
import subprocess
from pathlib import Path


app = typer.Typer()

class DownloadDataTask(object):
    """Download task Class
    It defines the environment variables:
        DATA_ROOT_DIR: Directory path to download the dataset
    Then executes the download script"""
    @staticmethod
    def run(data_dir: str) -> None:

        command = f"python 01_download_dataset.py --data_dir {data_dir}"
        splitted_command = command.split()
        process = subprocess.Popen(splitted_command, cwd=".")
        process.wait()

class DownloadModelTask(object):
    """Preprocess dataset task Class
    It defines the environment variables:
        DATA_ROOT_DIR: Dataset directory path
    Then executes the preprocess script"""
    @staticmethod
    def run(model_dir: str) -> None:

        env = os.environ.copy()
        env.update({
            'MODEL_DIR': model_dir,
        })

        process = subprocess.Popen("./download_model.sh", cwd=".", env=env)
        process.wait()

class InferTask(object):
    """Preprocess dataset task Class
    It defines the environment variables:
        DATA_DIR: Dataset directory path
        All other parameters are defined in the parameters_file
    Then executes the benchmark script"""
    @staticmethod
    def run(log_dir: str, data_dir: str, model_dir: str, out_dir) -> None:
        cmd = f"python3.7 chexpert.py --log_dir={log_dir} --data_dir={data_dir} --model_dir={model_dir} --out_dir={out_dir}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()

@app.command("download_data")
def download_data(data_dir: str = typer.Option(..., '--data_dir')):
    DownloadDataTask.run(data_dir)

@app.command("download_model")
def download_model(model_dir: str = typer.Option(..., '--model_dir')):
    DownloadModelTask.run(model_dir)

@app.command("infer")
def infer(log_dir: str = typer.Option(..., '--log_dir'),
          data_dir: str = typer.Option(..., '--data_dir'),
          model_dir: str = typer.Option(..., '--model_dir'),
          out_dir: str = typer.Option(..., '--out_dir')):
    InferTask.run(log_dir, data_dir, model_dir, out_dir)

if __name__ == '__main__':
    app()