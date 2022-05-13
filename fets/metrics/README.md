# FeTS Challenge - MLCube integration - Metrics

Challange repo: ["FeTS Instructions Repo"](https://github.com/FETS-AI/Challenge)

## Dataset

Please refer to the [FeTS challenge page](https://fets-ai.github.io/Challenge/data/) and follow the instructions.

## Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the examples from GitHub
git clone https://github.com/mlcommons/mlcube_examples
cd ./mlcube_examples/fets/model/mlcube
```

## Important files

These are the most important files on this project:

```bash

├── mlcube
│   ├── mlcube.yaml                             # MLCube configuration file, it defines the project, author, platform, docker and tasks.
│   └── workspace
│       ├── data
│       │   ├── ground_truth
│       │   │   └── BraTS_example_seg.nii.gz    # Ground truth example file
│       │   └── predictions
│       │       └── BraTS_example_seg.nii.gz    # Prediction example file
│       ├── parameters.yaml
│       └── results.yaml                        # Final output file containing result metrics.
└── project
    ├── Dockerfile                              # Docker file with instructions to create the image for the project.
    ├── metrics.py                              # Python file that contains the main logic of the project.
    ├── mlcube.py                               # Python entrypoint used by MLCube, contains the logic for MLCube tasks.
    └── requirements.txt                        # Python requirements needed to run the project inside Docker.
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
    * output_path: File path where output metrics will be stored

    This task takes the input predictions and ground truth data, perform the evaluation and then save the output result in the output_path.

### MLCube python file

The `mlcube.py` file is the handler file and entrypoint described in the dockerfile, here you can find all the logic related to how to process each MLCube task. If you want to add a new task first you must define it inside the `mlcube.yaml` file with its input and output parameters and then you need to add the logic to handle this new task inside the `mlcube.py` file.

### Metrics file

The `metrics.py` file contains the main logic of the project, you can modify this file and write your implementation here to calculate different metrics, this metrics file is called from the `mlcube.py` file and there are other ways to link your implementation and shown in the [MLCube examples repo](https://github.com/mlcommons/mlcube_examples).

## Tasks execution

```bash
# Run evaluate task.
mlcube run --mlcube=mlcube_cpu.yaml --task=evaluate
```

We are targeting pull-type installation, so MLCube images should be available on Docker Hub. If not, try this:

```Bash
mlcube run ... -Pdocker.build_strategy=always
```
