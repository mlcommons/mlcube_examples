# Packing an existing projecto into MLCUbe

In this tutorial we're going to use the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html). We'll take an existing implementation, create the needed files to pack it into MLCube and execute all tasks.


## Original project code

At fist we have only 4 files, one for package dependencies and 3 scripts for each task: download data, preprocess data and train.

```
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

We'll need some files for MLCube, first we'll need to create a folder called **mlcube** in the same path from as project folder. We'll need to create the following structure (for this tutorial the files are already in place but some of them are empty for you to define their content)

```
├── mlcube
│   ├── .mlcube.yaml
│   ├── platforms
│   │   └── docker.yaml
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

### MLCube handler Python file

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

```
├── mlcube
│   ├── .mlcube.yaml
│   ├── platforms
│   │   └── docker.yaml
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

The file located in **mlcube/.mlcube.yaml** contains the definition of all the tasks and their parameters.

This file is already provided, please take a look and study its content.

With this file we have finished the packing of the project into MLCube! Now we can setup the project and run all the tasks.


### Project setup
```Python
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/mlcommons/mlcube && cd ./mlcube
cd ./mlcube && python setup.py bdist_wheel  && pip install --force-reinstall ./dist/mlcube-* && cd ..
cd ./runners/mlcube_docker && python setup.py bdist_wheel  && pip install --force-reinstall --no-deps ./dist/mlcube_docker-* && cd ../../..
python3 -m pip install tornado

# Fetch the boston housing example from GitHub
git clone https://github.com/mlcommons/mlcube_examples && cd ./mlcube_examples
git fetch origin pull/27/head:feature/boston_housing && git checkout feature/boston_housing
cd ./boston_housing/project

# Build MLCube docker image.
docker build . -t mlcommons/boston_housing:0.0.1 -f Dockerfile

# Show tasks implemented in this MLCube.
cd ../mlcube && mlcube describe
```

### Dataset

The [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) will be downloaded and processed. Sizes of the dataset in each step:

| Dataset Step                   | MLCube Task       | Format     | Size    |
|--------------------------------|-------------------|------------|---------|
| Downlaod (Compressed dataset)  | download_data     | txt file   | ~52 KB  |
| Preprocess (Processed dataset) | preprocess_data   | csv file   | ~40 KB |
| Total                          | (After all tasks) | All        | ~92 KB |

### Tasks execution
```
# Download Boston housing dataset. Default path = /workspace/data
# To override it, use --data_dir=DATA_DIR
mlcube run --task download_data --platform docker

# Preprocess Boston housing dataset, this will convert raw .txt data to .csv format
# It will use the DATA_DIR path defined in the previous step
mlcube run --task preprocess_data --platform docker

# Run training.
# Parameters to override: --dataset_file_path=DATASET_FILE_PATH --parameters_file=PATH_TO_TRAINING_PARAMS
mlcube run --task train --platform docker
```