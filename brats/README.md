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
git fetch origin pull/xx/head:feature/brats && git checkout feature/brats
cd ./brats
```

## Tasks execution

```bash
# Run implementation with CPU support.
mlcube run --mlcube=mlcube_cpu.yaml --task=run

# Run implementation with GPU support.
mlcube run --mlcube=mlcube_gpu.yaml --task=run
```

We are targeting pull-type installation, so MLCube images should be available on Docker Hub. If not, try this:

```Bash
mlcube run ... -Pdocker.build_strategy=always
```
