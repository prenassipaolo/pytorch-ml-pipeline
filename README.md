# Pytorch ML Pipeline

This repository contains a starter implementation of a Machine Learning pipeline with Pytorch.

The goal of this implementation is to be simple, flexible, and easy to extend to your own projects. Its modular structure will allow to build complex and various models, without compromising the integrity and simplicity of the code.

Notice that this implementation is a work in progress -- currently we restrict to a classification problem architecture, along with its specific metrics and building blocks. But this is not a limitation: once the simplicity of the repository is grasped, extending it to additional tasks will be easy enough.

## Table of Contents

* [Overview](#overview)
    * [Generic Pipeline](#generic-pipeline)
    * [Pytorch Building Blocks](#pytorch-building-blocks)
    * [Folder Structure](#folder-structure)
* [Installation](#installation)
    * [Prerequisites](#prerequisites)
    * [Cloning](#cloning)
* [Usage](#usage)
    * [Modular Usage](#modular-usage)
    * [Generic Usage](#generic-usage)

## Overview

Each machine learning pipeline can be decomposed in pretty much the same building blocks. With this in mind, we propose a modular implementation where one single class (Model) captures all pipeline features:
- its attributes correspond to the overall procedure building blocks (model architectures, criterions, hyperparameters, ...)
- its methods consitst in the actions one can perform (preprocess, train, save, ...)

### Generic Pipeline

Usual machine learning workflow involves the following steps:

1. Load and preprocess data
2. Extract features
3. Split into training, validation, and test sets
4. Train model
5. Evaluate model
6. Tune hyperparameters
7. Test model
8. Collect results

### Pytorch Building Blocks

Pytorch is becoming the most used deep learning framework in academic research.

Typical deep learning implementation in Pytorch requires the following building blocks:
* model architecture
* criterion (loss, ..)
* optimizer

Additional regularization techniques can be added to avoid overfitting. In particular we implemented:
* early stopping
* learning rate scheduler


### Folder Structure
We created a modular structure, where each folder represents a different building block and contains its possible implementations. Then, to build the overall model the user has two options: 
- specify the module names in a json parameters file
- load the modules directly in the main script

```
.
├── README.md
├── data
│   └── download_data.py
├── main.py
├── model
│   ├── architectures
│   │   └── feedforwardnet.py
│   ├── earlystopping
│   │   └── earlystopping.py
│   ├── losses
│   │   └── binarycrossentropy.py
│   ├── model.py
│   ├── optimizers
│   │   └── adam.py
│   ├── parameters.json
│   ├── schedulers
│   │   └── steplr.py
│   └── train.py
├── outputs
│   ├── model.h5
│   └── model.json
└── requirements.txt
```

## Installation

### Prerequisites

Prerequisites can be installed through the requirements.txt file as below
```
$ pip install requirements.txt
```

### Cloning

Clone the repository to your local machine

```
$ git clone https://github.com/m-bini/pytorch-classification-pipeline
```

## Usage
-- to be finished --

### Modular Usage
-- to be finished --
Example with MNIST dataset:

```
# parameters path
parameters_path = "model/parameters.json"
# load training and validation datasets
train_set, test_set = download_mnist_datasets()

# create Model instance
M = Model(parameters_path=parameters_path)
# do training
M.do_train(train_set, test_set)
# save model weights and parameters
M.save(inplace=True)
M.history.plot() 

```

### Generic Usage
-- to be finished --
