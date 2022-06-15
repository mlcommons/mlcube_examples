# Matmul MLCube 

This example multiplies matrix a by matrix b, producing a * b.   
Both matrices, a and b, must be the same type.   
Matrix dimensions for matrix a and matrix b are specified by the user in file ./workspace/shapes.yaml  
The matrix dimensions must be >=2 and the 2 inner dimensions must specify valid matrix multiplication dimensions. 


## Create and initialize python environment
```
virtualenv -p python3.9 ./env && source ./env/bin/activate
```

## Install MLCube and MLCube runners
```
pip install mlcube mlcube-docker mlcube-singularity
``` 

## Clone MLCube examples and go to the mlcube_examples directory
```
git clone https://github.com/mlcommons/mlcube_examples && cd ./mlcube_examples
```

## MLCube checks if the system settings file exists each time it runs. If it does not exist, it will create it. Also, on every run MLCube finds new installed runners and updates system settings files. Run the following command to see if all runners are available: 
```
mlcube config --get runners
```

##  To see configuration parameters for the docker runner, run the following:
```
mlcube config --get platforms.docker
```

## Open the ~/mlcube.yaml file, find the section for the docker platform, and update values accordingly. Most likely only the docker parameter needs to be checked. To change runner parameters on a command line (when running MLCubes), use -Prunner.* pattern, e.g. -Prunner.build_strategy=auto or -Prunner.docker="sudo docker"

 

## Run Matmul MLCube on a local machine with Docker runner
```
mlcube --log-level=info run --mlcube=matmul --platform=docker --task=matmul
```
