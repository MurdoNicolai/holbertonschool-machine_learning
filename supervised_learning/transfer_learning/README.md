# Transfer Learning

## Master
By: Alexa Orrico, Software Engineer at Holberton School

## Resources

Read or watch:

- A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning
- Transfer Learning
- Transfer learning & fine-tuning

### Definitions to skim:

- Transfer learning

### References:

- Keras Applications
- Keras Datasets
- tf.keras.layers.Lambda
- tf.image.resize
- A Survey on Deep Transfer Learning

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
### General

- What is transfer learning?
- What is fine-tuning?
- What is a frozen layer? How and why do you freeze a layer?
- How to use transfer learning with Keras applications

## Requirements
### General

- Allowed editors: vi, vim, emacs
- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.8)
- Your files will be executed with numpy (version 1.19.2) and tensorflow (version 2.6)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A README.md file, at the root of the folder of the project, is mandatory
- Your code should use the pycodestyle style (version 2.6)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)' and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise noted, you are not allowed to import any module except `import tensorflow.keras as K`
- All your files must be executable
- The length of your files will be tested using `wc`

## Tasks

### 0. Transfer Knowledge

Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:

Keras packages a number of deep learning models alongside pre-trained weights into an applications module.

- You must use one of the applications listed in Keras Applications
- Your script must save your trained model in the current working directory as `cifar10.h5`
- Your saved model should be compiled
- Your saved model should have a validation accuracy of 87% or higher
- Your script should not run when the file is imported

In the same file, write a function `def preprocess_data(X, Y):` that pre-processes the data for your model:

- `X` is a `numpy.ndarray` of shape `(m, 32, 32, 3)` containing the CIFAR 10 data, where `m` is the number of data points
- `Y` is a `numpy.ndarray` of shape `(m,)` containing the CIFAR 10 labels for `X`
- Returns: `X_p, Y_p`
  - `X_p` is a `numpy.ndarray` containing the preprocessed `X`
  - `Y_p` is a `numpy.ndarray` containing the preprocessed `Y`
