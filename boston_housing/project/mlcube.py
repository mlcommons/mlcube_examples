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

class PreprocessDataTask(object):
    """Preprocess dataset task Class
    It defines the environment variables:
        DATA_ROOT_DIR: Dataset directory path
    Then executes the preprocess script"""
    @staticmethod
    def run(data_dir: str) -> None:

        command = f"python 02_preprocess_dataset.py --data_dir {data_dir}"
        splitted_command = command.split()
        process = subprocess.Popen(splitted_command, cwd=".")
        process.wait()

class TrainTask(object):
    """Preprocess dataset task Class
    It defines the environment variables:
        DATA_DIR: Dataset directory path
        All other parameters are defined in the parameters_file
    Then executes the benchmark script"""
    @staticmethod
    def run(dataset_file_path: str, parameters_file: str) -> None:
        with open(parameters_file, 'r') as stream:
            parameters = yaml.safe_load(stream)

        env = os.environ.copy()
        env.update({
            'DATASET_FILE_PATH': dataset_file_path,
        })

        env.update(parameters)

        process = subprocess.Popen("./run_and_time.sh", cwd=".", env=env)
        process.wait()

@app.command("download_data")
def download_data(data_dir: str = typer.Option(..., '--data_dir')):
    DownloadDataTask.run(data_dir)

@app.command("preprocess_data")
def preprocess_data(data_dir: str = typer.Option(..., '--data_dir')):
    PreprocessDataTask.run(data_dir)

@app.command("train")
def train(dataset_file_path: str = typer.Option(..., '--dataset_file_path'),
          parameters_file: str = typer.Option(..., '--parameters_file')):
    TrainTask.run(dataset_file_path, parameters_file)

if __name__ == '__main__':
    app()