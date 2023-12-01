# GPUs example

## Project setup

An important requirement is that you must have Docker and/or Singularity installed.

```bash
# Create Python environment and install MLCube with runners 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker mlcube-singularity
# Fetch the gpus example from GitHub
git clone https://github.com/mlcommons/mlcube_examples && cd ./mlcube_examples
git fetch origin pull/xxx/head:feature/gpu_example && git checkout feature/gpu_example
cd ./gpu_example/
```

## MLCube tasks

There is only one taks that will output the variable `CUDA_VISIBLE_DEVICES` along with the ouput of the `nvidia-smi` command:

```shell
mlcube run --task=check_gpus
```

You can modify the number of gpus by editing the number of `accelerator_count` inside the **mlcube.yaml** file.

Also you can override the number of gpus to use by using the `--gpus` flag when running the command, example:

```shell
mlcube run --task=check_gpus --gpus=2
```

### Singularity

For running on Singularity, you can define the platform while running the command as follows:

```shell
mlcube run --task=check_gpus --platform=singularity
```
