# MLCube + OpenFL: Mnist example

### Project setup
```Python
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/sergey-serebryakov/mlbox.git && cd mlbox && git checkout feature/configV2
cd ./mlcube && python setup.py bdist_wheel  && pip install --force-reinstall ./dist/mlcube-* && cd ..
cd ./runners/mlcube_docker && python setup.py bdist_wheel  && pip install --force-reinstall --no-deps ./dist/mlcube_docker-* && cd ../../..
```

## Clone MLCube examples and go to mnist_openfl directory
```
git clone https://github.com/mlperf/mlcube_examples.git && cd ./mlcube_examples
git fetch origin pull/33/head:feature/openfl && git checkout feature/openfl
cd ./mnist_openfl
```

## Run MNIST MLCube on a local machine with Docker runner
```
# Run MNIST training tasks: download data, train model and evaluate model
mlcube run --task download
mlcube run --task train
mlcube run --task evaluate
```

Parameters defined in **mlcube.yaml** can be overridden using: `param=input`, example:

```
python mlcube_cli.py run --task download data_dir=path_to_custom_dir
```

We are targeting pull-type installation, so MLCubes should be available on docker hub. If not, try this:

```
mlcube run ... -Pdocker.build_strategy=auto
```

By default, at the end of the train task, Mnist model will be saved in `workspace/model`.

By default, metrics for the train and evaluate task will be saved in `workspace/metrics`.
