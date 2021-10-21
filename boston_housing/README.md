# Packing an existing project into MLCUbe

In this tutorial we're going to use the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html). We'll take an existing implementation, create the needed files to pack it into MLCube and execute all tasks.

## Original project code

At fist we have only 4 files, one for package dependencies and 3 scripts for each task: download data, preprocess data and train.

```bash
├── project
    ├── 01_download_dataset.py
    ├── 02_preprocess_dataset.py
    ├── 03_train.py
    └── requirements.txt
```

The most important thing that we need to remember about these scripts are the input parameters:

* 01_download_dataset.py

**--data_dir** : Dataset download path, inside this folder path a new file called raw_dataset.txt will be created.

* 02_preprocess_dataset.py

**--data_dir** : Folder path containing raw dataset file, when finished a new file called processed_dataset.csv will be created.

* 03_train.py

**--dataset_file_path** : Processed dataset file path. Note: this is the full path to the csv file.
**--n_estimators** : Number of boosting stages to perform. In this case we're using a gradient boosting regressor.

## MLCube scructure

We'll need a couple of files for MLCube, first we'll need to create a folder called **mlcube** in the same path from as project folder. We'll need to create the following structure (for this tutorial the files are already in place)

```bash
├── mlcube
│   ├── mlcube.yaml
│   └── workspace
│       └── parameters.yaml
└── project
    ├── 01_download_dataset.py
    ├── 02_preprocess_dataset.py
    ├── 03_train.py
    └── requirements.txt
```

In the following steps we'll describe each file.

## Define tasks execution scripts

In general, we'll have a script for each task, and there are different ways to describe their execution from a main hanlder file, in this tutorial we'll use a function from the Python subprocess modeule:

* subprocess.Popen()

When we don't have input parameters for a Python script (or maybe just one) we can describe the execution of that script from Python code as follows:

```Python
import subprocess
# Set the full command as variable
command = "python my_task.py --single_parameter input"
# Split the command, this will give us the list:
# ['python', 'my_task.py', '--single_parameter', 'input']
splitted_command = command.split()
# Execute the command as a new process
process = subprocess.Popen(splitted_command, cwd=".")
# Wait for the process to finish
process.wait()
```

### MLCube File: mlcube/workspace/parameters.yaml

When we have a script with multiple input parameters, it will be hard to store the full command to execute it in a single variable, in this case we can create a shell script describing all the arguments and even add some extra fucntionalities, this will useful since we can define the input parameters as environment variables.

We can use the **mlcube/workspace/parameters.yaml** file to describe all the input parameters we'll use (this file is already provided, please take a look and study its content), the idea is to describe all the parameters in this file and then use this single file as an input for the task. Then we can read the content of the parameters file in Python and set all the parameters as environment variables. Finally with the environment variables setted we can excute a shell script with our implementation.

The way we execute all these steps in Python is described below.

```Python
import os
import yaml
# Read the file and store the parameters in a variable
with open(parameters_file, 'r') as stream:
    parameters = yaml.safe_load(stream)
# Get the system's enviroment
env = os.environ.copy()
# We can add a single new enviroment as follows
env.update({
'NEW_ENV_VARIABLE': "my_new_env_variable",
})
# Add all the parameters we got from the parameters file
env.update(parameters)
# Execute the shell script with the updated enviroment
process = subprocess.Popen("./run_and_time.sh", cwd=".", env=env)
# Wait for the process to finish
process.wait()
```

### Shell script

In this tutorial we already have a shell script containing the steps to run the train task, the file is: **project/run_and_time.sh**, please take a look and study its content.

### MLCube Command

We are targeting pull-type installation, so MLCube images should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=auto
```

Parameters defined in mlcube.yaml can be overridden using: param=input, example:

```bash
mlcube run --task=download_data data_dir=absolute_path_to_custom_dir
```

Also, users can override the workspace directory by using:

```bash
mlcube run --task=download_data --workspace=absolute_path_to_custom_dir
```

Note: Sometimes, overriding the workspace path could fail for some task, this is because the input parameter parameters_file should be specified, to solve this use:

```bash
mlcube run --task=train --workspace=absolute_path_to_custom_dir parameters_file=$(pwd)/workspace/parameters.yaml
```

### MLCube Python entrypoint file

At this point we know how to execute the tasks sripts from Python code, now we can create a file that contains the definition on how to run each task.

This file will be located in **project/mlcube.py**, this is the main file that will serve as the entrypoint to run all tasks.

This file is already provided, please take a look and study its content.

## Dockerize the project

We'll create a Dockerfile with the needed steps to run the project, at the end we'll need to define the execution of the **mlcube.py** file as the entrypoint. This file will be located in **project/Dockerfile**.

This file is already provided, please take a look and study its content.

When creating the docker image, we'll need to run the docker build command inside the project folder, the command that we'll use is:

`docker build . -t mlcommons/boston_housing:0.0.1 -f Dockerfile`

Keep in mind the tag that we just described.

At this point our solution folder structure should look like this:

```bash
├── mlcube
│   ├── mlcube.yaml
│   └── workspace
│       └── parameters.yaml
└── project
    ├── 01_download_dataset.py
    ├── 02_preprocess_dataset.py
    ├── 03_train.py
    ├── Dockerfile
    ├── mlcube.py
    ├── requirements.txt
    └── run_and_time.sh
```

### Define MLCube files

Inside the mlcube folder we'll need to define the following files.

### mlcube/platforms/docker.yaml

This file contains the description of the platform that we'll use to run MLCube, in this case is Docker. In the container definition we'll have the following subfields:

* command: Main command to run, in this case is docker
* run_args: In this field we'll define all the arguments to run the docker conatiner, e.g. --rm, --gpus, etc.
* image: Image to use, in this case we'll need to use the same image tag from the docker build command.

This file is already provided, please take a look and study its content.

### MLCube task definition file

The file located in **mlcube/mlcube.yaml** contains the definition of all the tasks and their parameters.

This file is already provided, please take a look and study its content.

With this file we have finished the packing of the project into MLCube! Now we can setup the project and run all the tasks.

### Project setup

## Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the boston housing example from GitHub
git clone https://github.com/mlcommons/mlcube_examples && cd ./mlcube_examples
git fetch origin pull/27/head:feature/boston_housing && git checkout feature/boston_housing
cd ./boston_housing/mlcube
```

### Dataset

The [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) will be downloaded and processed. Sizes of the dataset in each step:

| Dataset Step                   | MLCube Task       | Format     | Size    |
|--------------------------------|-------------------|------------|---------|
| Downlaod (Compressed dataset)  | download_data     | txt file   | ~52 KB  |
| Preprocess (Processed dataset) | preprocess_data   | csv file   | ~40 KB |
| Total                          | (After all tasks) | All        | ~92 KB |

### Tasks execution

```bash
# Download Boston housing dataset. Default path = /workspace/data
# To override it, use data_dir=DATA_DIR
mlcube run --task download_data

# Preprocess Boston housing dataset, this will convert raw .txt data to .csv format
# It will use the DATA_DIR path defined in the previous step
mlcube run --task preprocess_data

# Run training.
# Parameters to override: dataset_file_path=DATASET_FILE_PATH parameters_file=PATH_TO_TRAINING_PARAMS
mlcube run --task train
```
