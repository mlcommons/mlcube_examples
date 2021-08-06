# MLCube™ examples

The machine learning (ML) community has seen an explosive growth and innovation in the last decade. New models emerge 
on a daily basis, but sharing those models remains an ad-hoc process. Often, when a researcher wants to use a model 
produced elsewhere, they must waste hours or days on a frustrating attempt to get the model to work. Similarly, a ML 
engineer may struggle to port and tune models between development and production environments which can be significantly 
different from each other. This challenge is magnified when working with a set of models, such as reproducing related 
work, employing a performance benchmark suite like MLPerf, or developing model management infrastructures. 
Reproducibility, transparency and consistent performance measurement are cornerstones of good science and engineering. 

The field needs to make sharing models simple for model creators, model users, developers and operators for both 
experimental and production purpose while following responsible practices. Prior works in the MLOps space have provided 
a variety of tools and processes that simplify user journey of deploying and managing ML in various environments, 
which include management of models, datasets, and dependencies, tracking of metadata and experiments, deployment and 
management of ML lifecycles, automation of performance evaluations and analysis, etc.

We propose an MLCube, a contract for packaging ML tasks and models that enables easy sharing and consistent reproduction 
of models, experiments and benchmarks amidst these existing MLOps processes. MLCube differs from an operation tool by 
acting as a contract and specification as opposed to a product or implementation. 

This repository contains a number of MLCube examples that can run in different environments using 
[MLCube runners](https://github.com/mlperf/mlcube). 

1. [MNIST](./mnist) MLCube downloads data and trains a simple neural network. This MLCube can run with Docker or
   Singularity locally and on remote hosts. The [README](./mnist/README.md) file provides instructions on how to run it.
   MLCube [documentation](https://mlperf.github.io/mlcube/getting-started/mnist/) provides additional details. 
2. [Hello World](./hello_world) MLCube is a simple exampled described in this 
   [tutorial](https://mlperf.github.io/mlcube/getting-started/hello-world/).
3. [EMDenoise](./emdenoise) MLCube downloads data and trains a deep convolutional neural network
   for Electron Microscopy Benchmark. This MLCube can only run the Docker container.
   The [README](./emdenoise/README.md) file provides instructions on how to run it.
4. [Matmul](./matmul) Matmul performs a matrix multiply. 
