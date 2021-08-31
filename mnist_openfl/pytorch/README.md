# MLCube + OpenFL: Mnist example - Pytorch

## Project setup

```bash
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository
# (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/mlcommons/mlcube && cd mlcube/mlcube
python setup.py bdist_wheel  && pip install --force-reinstall ./dist/mlcube-* && cd ..
cd ./runners/mlcube_docker && python setup.py bdist_wheel  && pip install --force-reinstall --no-deps ./dist/mlcube_docker-* && cd ../../..
```

## Clone MLCube examples and go to mnist_openfl directory

```bash
git clone https://github.com/mlperf/mlcube_examples.git && cd ./mlcube_examples
git fetch origin pull/33/head:feature/openfl && git checkout feature/openfl
cd ./mnist_openfl/pytorch
```

## Run MNIST MLCube on a local machine with Docker runner

```bash
# Run MNIST training tasks: download data, train model and evaluate model
mlcube run --task download
mlcube run --task train
mlcube run --task evaluate
```

We are targeting pull-type installation, so MLCubes should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=auto
```

Parameters defined in **mlcube.yaml** can be overridden using: `param=input`, example:

```bash
mlcube run --task=download data_dir=absolute_path_to_custom_dir
```

Also, users can override the workspace directory by using:

```bash
mlcube run --task=download --workspace=absolute_path_to_custom_dir
```

**Note:** Sometimes, overriding the workspace path may fail when running train and evaluate task, this is because the input parameter *parameters_file* should be specified, to solve this use:

```bash
mlcube run --task=train --workspace=absolute_path_to_custom_dir parameters_file=$(pwd)/workspace/parameters/default.parameters.yaml
```

By default, at the end of the train task, Mnist model will be saved in `workspace/model`.

By default, metrics for the train and evaluate task will be saved in `workspace/metrics`.
