# MNIST MLCube

Proof-of-concept for MLCube configuration 2.0. Please, review the following:

1. Local user configuration for docker (and other) runtime: [.mlcube.yaml](../.mlcube.yaml). This file here is for
   example, and is not shipped with MLCubes in general. It contains common options for MLCube execution environments
   common to all MLCubes. 
2. New MLCube configuration file 2.0 [mlcube.yaml](../mlcube.py)
3. The [mlcube.py](../mlcube.py) here simulates MLCube CLI. It is temporary stored here, and is part of MLCube library.

This modified MNIST example depends on latest version of docker runner located in this branch:
```
https://github.com/sergey-serebryakov/mlbox/tree/feature/configV2
```

I do not think it's worth spending time now trying to reproduce results, but in case it's required:
1. Clone that branch.
2. Create virtual environment with python >= 3.6.
3. Export PYTHONPATH variable. Only mlcube_docker is requried to be present.
4. Install mlcube dependencies (mlcube/requirements.txt) and omegaconf
5. Run this example:
   ```bash
   python mlcube.py show_config --mlcube=./mnist --platform=docker --resolve
   python mlcube.py run --mlcube ./mnist --task download --platform docker
   python mlcube.py run --mlcube ./mnist --task train --platform docker
   ```

The example implementation uses OmegaConf, so users can use configuration variables:
```shell
$ mlcube run ... --workspace=/nfs/workspace/mnist
# workspace: /nfs/workspace/mnist

$ mlcube run ... --workspace='/nfs/workspace/${name}'
# workspace: /nfs/workspace/mnist

$ mlcube run ... --workspace='~/.mlcube/workspace/${name}'
# workspace: /home/developer/.mlcube/workspace/mnist
```
