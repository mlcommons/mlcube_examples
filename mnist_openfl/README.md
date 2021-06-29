# MNIST MLCube

## Create and initialize python environment
```
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker tornado
```

## Clone MLCube examples and go to MNIST root directory
```
git clone https://github.com/mlperf/mlcube_examples.git && cd ./mlcube_examples
git fetch origin pull/26/head:feature/openfl && git checkout feature/openfl
cd ./mnist_openfl
```

## Run MNIST MLCube on a local machine with Docker runner
```
# Configure MNIST MLCube
mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml

# Run MNIST training tasks: download data and train the model
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/download.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/train.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/evaluate.yaml
```

At the end of the train task, Mnist model will be saved in `workspace/model`.

Metrics for the train and evaluate task will be saved in `workspace/metrics`.