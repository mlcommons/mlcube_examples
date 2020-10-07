# MNIST MLCommons-Box

```
# Clone MLCommons-Box Examples
git clone https://github.com/mlperf/box_examples.git && cd ./box_examples

# Create Python Virtual Environment
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCommons-Box Docker runner that is used by MNIST 
pip install mlcommons-box-docker

# Go inside MNIST MLCommons-Box directory. 
cd ./mnist

# Run 'download' task.
mlcommons_box_docker run --mlbox=. --platform=./platforms/docker.yaml --task=./run/download.yaml

# Run 'train' task.
mlcommons_box_docker run --mlbox=. --platform=./platforms/docker.yaml --task=./run/train.yaml
```
