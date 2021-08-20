# MLCube: Chexpert Example
This example demonstrates how to use MLCube to work with a computer vision model trained on the CheXpert Dataset. 

CheXpert is a large dataset of chest X-rays and competition for automated chest x-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets.

The model used here is based on the top 1 solution of the CheXpert challenge, which can be found [here](https://github.com/jfhealthcare/Chexpert).

### Project setup
```Python
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/sergey-serebryakov/mlbox.git && cd mlbox && git checkout feature/configV2
cd ./mlcube && python setup.py bdist_wheel  && pip install --force-reinstall ./dist/mlcube-* && cd ..
cd ./runners/mlcube_docker && python setup.py bdist_wheel  && pip install --force-reinstall --no-deps ./dist/mlcube_docker-* && cd ../../..
```

## Clone MLCube examples and go to chexpert
```
git clone https://github.com/mlperf/mlcube_examples.git && cd ./mlcube_examples
git fetch origin pull/XX/head:chest-xray-example && git checkout chest-xray-example
cd ./chexpert
```

## Get the data
Because the Chexpert Dataset contains sensitive information, signing an user agreement is required before obtaining the data. This means that we cannot automate the data download process. To obtain the dataset:

1. sign up at the [Chexpert Dataset Download Agreement](https://stanfordmlgroup.github.io/competitions/chexpert/#agreement) and download the small dataset from the link sent to your email.
2. Unzip and place the `CheXpert-v1.0-small` folder inside  `mlcube/workspace/data` folder. Your folder structure should look like this:
   
```
.
├── mlcube
│   └── workspace
│       └── Data 
│           └── CheXpert-v1.0-small
│          		├── valid
│          		└── valid.csv
└── project
```

## Run Chexpert MLCube on a local machine with Docker runner
```
# Run Chexpert training tasks: download data, train model and evaluate model
mlcube run --task download_model
mlcube run --task preprocess
mlcube run --task infer
```

Parameters defined in **mlcube.yaml** can be overridden using: `param=input`, example:

```
mlcube run --task download_model data_dir=path_to_custom_dir
```

We are targeting pull-type installation, so MLCubes should be available on docker hub. If not, try this:

```
mlcube run ... -Pdocker.build_strategy=auto
```

By default, at the end of the download_model task, Chexpert model will be saved in `workspace/model`.

By default, at the end of the infer task, results will be saved in `workspace/inferences.txt`.