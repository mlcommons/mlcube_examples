# Matmul MLCube 

This example multiplies matrix a by matrix b, producing a * b.   
Both matrices, a and b, must be the same type.   
Matrix dimensions for matrix a and matrix b are specified by the user in file ./workspace/shapes.yaml  
The matrix dimensions must be >=2 and the 2 inner dimensions must specify valid matrix multiplication dimensions. 


## Create and initialize python environment
```
virtualenv -p python3 ./env && source ./env/bin/activate 
```

## Install MLCube and MLCube runners
```
pip install mlcube mlcube-docker mlcube-singularity mlcube-ssh
``` 

## Clone MLCube examples and go to matmul root directory
```
git clone https://github.com/mlperf/mlcube_examples.git && cd ./mlcube_examples/matmul
```

## Run Matmul MLCube on a local machine with Docker runner
```
# Configure Matmul MLCube
mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml

# Run Matmul tasks: You can change the shapes of the matrices that are multipled in workspace/shapes.yaml 
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/matmul.yaml
```
Go to `workspace/` directory and study its content. Then:
```
ls ./workspace
cat ./workspace/matmul.txt
```
