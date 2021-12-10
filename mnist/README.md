# MNIST MLCube

This example MLCube trains a simple neural network using MNIST dataset. Concretely, it implements two tasks:

- `download` task downloads MNIST dataset.
- `train` trains a DL model.

```shell
# Create python virtual environment
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker/singularity runners
pip install mlcube mlcube-docker mlcube-singularity

# Show installed MLCube runners
mlcube config --get runners

# Show platform configurations. A platform is a configured instance of a runner.
mlcube config --get platforms

# Clone MLCube examples and go to MNIST root directory
git clone https://github.com/mlcommons/mlcube_examples.git && cd ./mlcube_examples/mnist

# Show MLCube overview
mlcube describe --mlcube .

# Show MLCube and MLCube docker and singularity configurations
mlcube show_config --resolve --mlcube . --platform docker
mlcube show_config --resolve --mlcube . --platform singularity

# Download data and train a model using default docker platform.
mlcube run --mlcube . --task download --platform docker
mlcube run --mlcube . --task train --platform docker
```
