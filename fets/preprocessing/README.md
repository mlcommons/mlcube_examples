# FeTS Challenge - MLCube integration - Preprocess

Challange repo: ["FeTS Instructions Repo"](https://github.com/FETS-AI/Challenge)

## Dataset

Please refer to the [FeTS challenge page](https://fets-ai.github.io/Challenge/data/) and follow the instructions.

## Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the boston housing example from GitHub
git clone https://github.com/mlcommons/mlcube_examples && cd ./mlcube_examples
git fetch origin pull/39/head:feature/fets && git checkout feature/fets
cd ./fets/preprocessing/mlcube
```

## Important files

These are the most important files on this project:

```bash

├── mlcube
│   ├── mlcube.yaml                             # MLCube configuration file, it defines the project, author, platform, docker and tasks.
│   └── workspace
│       ├── data
│       │   └── BraTS_example_seg.nii.gz        # Input data
│       ├── results
│       │   └── output.npy                      # Output processed data
│       ├── parameters.yaml
└── project
    ├── Dockerfile                              # Docker file with instructions to create the image for the project.
    ├── preprocess.py                           # Python file that contains the main logic of the project.
    ├── mlcube.py                               # Python entrypoint used by MLCube, contains the logic for MLCube tasks.
    └── requirements.txt                        # Python requirements needed to run the project inside Docker.
    └── run.sh                                  # Bash file containing logic to call preprocess.py script.
```

## How to modify this project

You can change each file described above in order to add your own implementation.

### Requirements file

In this file (`requirements.txt`) you can add all the python dependencies needed for running your implementation, these dependencies will be installed during the creation of the docker image, this happens when you run the ```mlcube run ...``` command.

### Dockerfile

You can use both, CPU or GPU version for the dockerfile (`Dockerfile_CPU`, `Dockerfile_GPU`), also, you can add or modify any steps inside the file, this comes handy when you need to install some OS dependencies or even when you want to change the base docker image, inside the file you can find some information about the existing steps.

### Parameters file

This is a yaml file (`parameters.yaml`)that contains all extra parameters that aren't files or directories, for example, here you can place all the hyperparameters that you will use for training a model. This file will be passed as an **input parameter** in the MLCube tasks and then it will be read inside the MLCube container.

### MLCube yaml file

In this file (`mlcube.yaml`) you can find the instructions about the docker image and platform that will be used, information about the project (name, description, authors), and also the tasks defined for the project.

In the existing implementation you will find 1 task:

* evaluate:

    This task takes the following parameters:

  * Input parameters:
    * predictions: Folder path containing predictions
    * ground_truth: Folder path containing ground truth data
    * parameters_file: Extra parameters
  * Output parameters:
    * output_path: File path where output preprocess will be stored

    This task takes the input predictions and ground truth data, perform the evaluation and then save the output result in the output_path.

### MLCube python file

The `mlcube.py` file is the handler file and entrypoint described in the dockerfile, here you can find all the logic related to how to process each MLCube task. If you want to add a new task first you must define it inside the `mlcube.yaml` file with its input and output parameters and then you need to add the logic to handle this new task inside the `mlcube.py` file.

### Preprocess file

The `preprocess.py` file contains the main logic of the project, you can modify this file and write your implementation here to perform the different preprocessing steps, this preprocess file is called from the `run.sh` file and there are other ways to link your implementation and shown in the [MLCube examples repo](https://github.com/mlcommons/mlcube_examples).

### Run bash file

The `run.sh` file is called from `mlcube.py` and it receives the arguments, here we can perform different steps to then call the `preprocess.py` script.

## Tasks execution

```bash
# Run preprocess task.
mlcube run --mlcube=mlcube_cpu.yaml --task=preprocess
```

To use Singularity runner add the flag `--platform=singularity`, example:

```bash
mlcube run --mlcube=mlcube.yaml --task=preprocess --platform=singularity
```

We are targeting pull-type installation, so MLCube images should be available on Docker Hub. If not, try this:

```Bash
mlcube run ... -Pdocker.build_strategy=always
```
