# MNIST MLCube

## Create and initialize python environment
```
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
```

## Clone MLCube examples and go to MNIST root directory
```
git clone https://github.com/mlperf/mlcube_examples.git && cd ./mlcube_examples/mnist
```

## Run MNIST MLCube on a local machine with Docker runner
```
# Configure MNIST MLCube
mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml

# Run MNIST training tasks: download data and train the model
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/download.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/train.yaml
```
Go to `workspace/` directory and study its content. 


## Run MNIST MLCube on a local machine with Singularity runner
```
# Configure MNIST MLCube
mlcube_singularity configure --mlcube=. --platform=platforms/singularity.yaml

# Run MNIST training tasks: download data and train the model
mlcube_singularity run --mlcube=. --platform=platforms/singularity.yaml --task=run/download.yaml
mlcube_singularity run --mlcube=. --platform=platforms/singularity.yaml --task=run/train.yaml
```
Go to `workspace/` directory and study its content. 


## Run MNIST MLCube on a remote machine with SSH runner
Setup passwordless access to a remote machine. Create and/or update your SSH configuration file (`~/.ssh/config`).
Create an alias for your remote machine. This will enable access for tools like `ssh`, `rsync` and `scp` using 
`mlcube-remote` name instead of actual name or IP address. 
```
Host mlcube-remote
    HostName {{IP_ADDRESS}}
    User {{USER_NAME}}
    IdentityFile {{PATH_TO_IDENTITY_FILE}}
```
Remove results of previous runs. Remove all directories in `workspace/` except `workspace/parameters`.

```
# Configure MNIST MLCube
mlcube_ssh configure --mlcube=. --platform=platforms/ssh.yaml

# Run MNIST training tasks: download data and train the model
mlcube_ssh run --mlcube=. --platform=platforms/ssh.yaml --task=run/download.yaml
mlcube_ssh run --mlcube=. --platform=platforms/ssh.yaml --task=run/train.yaml
```
Go to `workspace/` directory and study its content.
