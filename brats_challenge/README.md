# BraTS Challenge 2020 - MLCube integration

Original implementation: [open_brats2020](https://github.com/lescientifik/open_brats2020)

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

## Task execution

```bash
# Run training.
mlcube run --mlcube=mlcube_cpu.yaml --task=train

# Run inference.
mlcube run --mlcube=mlcube_gpu.yaml --task=inference
```

We are targeting pull-type installation, so MLCube images should be available on Docker Hub. If not, try this:

```Bash
mlcube run ... -Pdocker.build_strategy=always
```
