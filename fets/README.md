# MedPerf - MLCube - Federated Tumor Segmentation Challenge

This repository contains three [MLCube&reg;](https://github.com/mlcommons/mlcube) projects used throughout the organization of the FeTS challenge:

1. Preprocessing: This MLCube contains the preprocessing pipeline employed in FeTS (which is identical to BraTS).
2. Model: This MLCube can be used by **task-2 competitors** to build their inference application. If you are looking for how to prepare your FeTS task-2 submission, please continue reading [here](model).
3. Metrics: This MLCube computes the segmentation metrics established in BraTS for (prediction, reference segmentation)-pairs.

## Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the examples from GitHub
git clone https://github.com/mlcommons/mlcube_examples
cd ./mlcube_examples/fets/model/mlcube
```

## Execute docker-based MLCubes with Singularity runner

First, install the latest version of the Singularity runner.

```bash
virtualenv -p python3 env && source ./env/bin/activate

git clone https://github.com/mlcommons/mlcube && cd ./mlcube

git fetch origin pull/241/head:feature/singularity_with_docker_images && git checkout feature/singularity_with_docker_images

pip install semver spython && pip install ./mlcube

pip install --no-deps --force-reinstall ./runners/mlcube_singularity
```

* To convert a Docker image hosted in DockerHub to Singularity, please specify the Docker image name with its tag inside the **mlcube.yaml** file, example:

```yaml
docker:
  # Image name.
  image: my_user/my_image:0.0.1
```

* To convert a local Docker image, first find the Docker image ID:

```bash
docker images
#output:
#REPOSITORY        TAG       IMAGE ID        CREATED        SIZE
#my_image          latest    bf756fb1ae65    5 months ago   13.3kB
```

Then, create a tarball of the Docker image using the image ID:

```bash
docker save bf756fb1ae65 -o my_docker_image.tar 
```

Then, you must specify the path to the tarball file inside the **mlcube.yaml** file:

```yaml
docker:
  # Image name.
  image: my_user/my_image:0.0.1
  tar_file: path_to/my_docker_image.tar
```

After getting the **mlcube.yaml** done with the needed data from the Docker image you want to convert, you can convert the image while running any MLCube task using the following command:

```bash
mlcube run --mlcube=mlcube.yaml --task=my_task --platform=singularity
```

## MedPerf API Server

To run locally, clone this repo:

```Bash
git clone https://github.com/mlcommons/medperf.git
```

Go to the `server` folder

```Bash
cd server
```

Install all dependencies

```Bash
pip install -r requirements.txt
```

Create .env file with your environment settings

Sample .env.example is added to root. Rename `.env.example` to `.env` and modify with your env vars.

```Bash
cp .env.example .env
```

Create tables and existing models

```Bash
python manage.py migrate
```

Start the server

```Bash
python manage.py runserver
```

API Server is running at `http://127.0.0.1:8000/` by default. You can view and experiment Medperf API at `http://127.0.0.1:8000/swagger`

## Medperf CLI

The Medperf CLI is a command-line-interface that provides tools for preparing datasets and executing benchmarks on such datasets.

To install, clone this repo (If you already did skip this step):

```Bash
git clone https://github.com/mlcommons/medperf.git
```

Go to the `cli` folder

```Bash
cd cli
```

Install using pip

```Bash
pip install -e .
```

## How to run

The MedPerf CLI provides the following commands:

- `login`: authenticates the CLI with the medperf backend server

```Bash
medperf login
```

- `dataset ls`: Lists all registered datasets by the user

```Bash
medperf dataset ls
```

- `dataset create`: Prepares a raw dataset for a specific benchmark

```Bash
medperf dataset create -b <BENCHMARK_UID> -d <DATA_PATH> -l <LABELS_PATH>
```

- `dataset submit`: Submits a prepared local dataset to the platform.

```Bash
medperf dataset submit -d <DATASET_UID> 
```

- `dataset associate`: Associates a prepared dataset with a specific benchmark

```Bash
medperf associate -b <BENCHMARK_UID> -d <DATASET_UID>
```

- `run`: Alias for `result create`. Runs a specific model from a benchmark with a specified prepared dataset

```Bash
medperf run -b <BENCHMARK_UID> -d <DATASET_UID> -m <MODEL_UID>
```

- `result ls`: Displays all results created by the user

```Bash
medperf result ls
```

- `result create`: Runs a specific model from a benchmark with a specified prepared dataset

```Bash
medperf result create -b <BENCHMARK_UID> -d <DATASET_UID> -m <MODEL_UID>
```

- `result submit`: Submits already obtained results to the platform

```Bash
medperf result submit -b <BENCHMARK_UID> -d <DATASET_UID> -m <MODEL_UID>
```

- `mlcube ls`: Lists all mlcubes created by the user. Lists all mlcubes if `--all` is passed

```Bash
medperf mlcube ls [--all]
``` 

- `mlcube submit`: Submits a new mlcube to the platform

```Bash
medperf mlcube submit
```

- `mlcube associate`: Associates an MLCube to a benchmark

```Bash
medperf mlcube associate -b <BENCHMARK_UID> -m <MODEL_UID>
```

The CLI runs MLCubes behind the scene. This cubes require a container engine like docker, and so that engine must be running before running commands like `prepare` and `execute`
