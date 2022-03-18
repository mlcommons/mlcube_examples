# MedPerf - MLCube - BraTs Challange Integration

## Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the boston housing example from GitHub
git clone https://github.com/mlcommons/mlcube_examples && cd ./mlcube_examples
git fetch origin pull/39/head:feature/brats && git checkout feature/brats
cd ./brats/metrics/mlcube
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
