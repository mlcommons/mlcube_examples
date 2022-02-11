# BraTS Challenge 2020 - MLCube integration

Original implementation: ["BraTS Instructions Repo"](https://github.com/BraTS/Instructions)

## Dataset

Please refer to the [BraTS challenge page](http://braintumorsegmentation.org/) and follow the instructions in the data section.

## Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the boston housing example from GitHub
git clone https://github.com/mlcommons/mlcube_examples && cd ./mlcube_examples
git fetch origin pull/39/head:feature/brats && git checkout feature/brats
cd ./brats
```

## Important files

These are the most important files on this project:

```bash
├── Dockerfile_CPU       # Docker file with instructions to create the image for the CPU version.
├── Dockerfile_GPU      # Docker file with instructions to create the image for the GPU version.
├── mlcube.py            # Python entrypoint used by MLCube, contains the logic for MLCube tasks.
├── mlcube_cpu.yaml      # MLCube CPU configuration, defines the project, author, platform, docker and tasks.
├── mlcube_gpu.yaml      # MLCube GPU configuration, here the difference is the target dockerfile.
├── requirements.txt     # Python requirements needed to run the project inside Docker.
├── src                     
│   ├── my_logic.py      # Python file that contains the main logic of the project.
│   └── utils   
│       └── utilities.py # Python utilities file that stores useful functions.
└── workspace
    └── parameters.yaml  # File containing all extra parameters.
```

## Project workflow

![MLCube workflow](https://i.imgur.com/qXRp3Tb.png)

## How to modify this project

You can change each file described above in order to add your own implementation.

### Requirements file

In this file (`requirements.txt`) you can add all the python dependencies needed for running your implementation, these dependencies will be installed during the creation of the docker image, this happens when you run the ```mlcube run ...``` command.
### Dockerfile

You can use both, CPU or GPU version for the dockerfile (`Dockerfile_CPU`, `Dockerfile_GPU`), also, you can add or modify any steps inside the file, this comes handy when you need to install some OS dependencies or even when you want to change the base docker image, inside the file you can find some information about the existing steps.


### Parameters file

This is a yaml file (`parameters.yaml`)that contains all extra parameters that aren't files or directories, for example, here you can place all the hyperparameters that you will use for training a model. This file will be passed as an **input parameter** in the MLCube tasks and then it will be read inside the MLCube container.

### MLCube yaml file

In these files (`mlcube_cpu`, `mlcube_gpu`) you can find the instructions about the docker image and platform that will be used, information about the project (name, description, authors), and also the tasks defined for the project.

In the existing implementation you will find 2 tasks:

* example:

    It only takes one input parameter: parameters file.
    This task reads one specific parameter from the parameters file () and then prints the value of the parameter.

* run:

    This task takes the following parameters:

    * Input parameters:
        * input_folder: folder path containing input data
        * parameters_file: Extra parameters
    * Output parameters:
        * output_folder: folder path where output data will be stored
    
    This task takes the input data, "process it" and then save the output result in the output_folder, it also prints some information from the extra parameters.


### MLCube python file

The `mlcube.py` file is the handler file and entrypoint described in the dockerfile, here you can find all the logic related to how to process each MLCube task. If you want to add a new task first you must define it inside the `mlcube.yaml` file with its input and output parameters and then you need to add the logic to handle this new task inside the `mlcube.py` file.

### Main logic file

The `my_logic.py` file contains the main logic of the project, you can modify this file and write your implementation here, this logic file is called from the `mlcube.py` file and there are other ways to link your implementation and shown in the [MLCube examples repo](https://github.com/mlcommons/mlcube_examples).

### Utilities file

In the `utilities.py` file you can add some functions that will be useful for your main implementation, in this case, the functions from the utilities file are used inside the main logic file.
## Tasks execution

```bash
# Run example task with CPU support.
mlcube run --mlcube=mlcube_cpu.yaml --task=example

# Run main task with CPU support.
mlcube run --mlcube=mlcube_cpu.yaml --task=run

# Run example task with GPU support.
mlcube run --mlcube=mlcube_gpu.yaml --task=example

# Run main task with GPU support.
mlcube run --mlcube=mlcube_gpu.yaml --task=run
```

We are targeting pull-type installation, so MLCube images should be available on Docker Hub. If not, try this:

```Bash
mlcube run ... -Pdocker.build_strategy=always
```
