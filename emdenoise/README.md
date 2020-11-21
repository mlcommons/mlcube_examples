# EMDenoise MLCube

## EMDenoise: Electron Microscopy Benchmark

Authors: Patrick Austin (RAL, STFC), Keith Butler (SciML, STFC)*, Jeyan Thiyagalingam*, and Tony Hey.
*(Corresponding)

### Overview

Electron microscopy has been undergoing revolution, with the advent of direct counting electron
detectors providing very high detective quantum efficiency and low readout noise, facilitating
very rapid frame collection rates.

### Uses

Increasingly the importance of machine learning techniques is being recognised and exploited in
materials’ science, for example identifying microstructure. Increased frame collection rates on
transmission electron microscopes (TEM) allows the observation of dynamic processes such as defect
migration and surface reconstruction. In these scenarios images are collected at a high frequency,
machine learning techniques for rapid identification of features and objects within the image, such
as semantic segmentation are already powerful tools.

### Impact

Rapid machine learning, facilitated analysis and processing of images offers the promise of
microscopes which automatically optimise data acquisition, or act as a monitor in nano-fabrication,
or for alerting microscope operators of potentially important events.

![Deep learning to denoise electron microscopy. A noisy image collected by the microscope is
passed through a neural network which produces a new image with greatly improved signal to noise
ratio.](fig1.jpg)

### Applications

In almost all instances it is desirable to have techniques to improve signal to noise ratios. For
example, being able to image at lower electron doses can facilitate experiments with reduced beam
induced phenomena taking place in samples; however these images are inevitably noisier than at
higher doses. Denoising can facilitate low-dose experiments, with image quality comparable to high-
dose experiments. Greater time resolution can be achieved with the aid of effective image denoising
procedures.

### Previous work

We applied different deep-learning architectures in denoising TEM images of defective graphene
sheets, assessing the performance in terms of three key considerations when deploying a denoising
workflow:

1. fidelity of the denoised image;
2. speed of the algorithm;
3. training data requirements

We are able to achieve denoising performance far in advance of the best available classical denoising
techniques, opening the door for the application of deep-learning denoisng to facilitate enhanced
electron microscopy.

![Comparing the performance of deep-learning techniques. Left a noisy microscope image,
right the best classical denoising procedure produces a very low-detail result, centre a range of deep-
learning techniques all produce much clearer results than the classical method allowing the
identification of atomic level defects in the material.](fig2.jpg)

## Create and initialize python environment
```
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
```

## Clone MLCube examples and go to EMDenoise root directory
```
git clone https://github.com/mlperf/mlcube_examples.git && cd ./mlcube_examples/emdenoise
```

## Run EMDenoise MLCube on a local machine with Docker runner
```
# Configure EMDenoise MLCube
mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml

# Run EMDenoise tasks: download data, preprocess data train and test the model
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/download.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/preprocess.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/train.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/test.yaml
```
Go to `workspace/` directory and study its content.

## Contribution

Contribution by Digital Science Center, Indiana University Bloomington.

## References

1. https://github.com/stfc-sciml/sciml-benchmarks