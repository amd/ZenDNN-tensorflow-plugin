# zentf C++ Examples
This document provides examples for running inference on CNN models using zentf C++ interface.

## Table of Contents
- [Set-up zentf for C++ user interface](#set_up-zentf-for-c_user-interface)
- [Build C++ inference application](#build-c_inference-application)
- [CNN examples](#cnn-examples)

## Set-up zentf for C++ user interface

Please follow the instructions in the [README.md](../../scripts/c%2B%2B/README.md) file for setting up zentf.

## Build C++ inference application

### 1. Compile the sample inference application
```
$ cd <cpp_package_name>/
$ g++ examples/sample_inference.cpp -o sample_inference -I./<tf_folder>/tensorflow/include -L./<tf_folder>/tensorflow/ -ltensorflow_framework -ltensorflow_cc -Wl,-rpath=./<tf_folder>/tensorflow/ -std=c++17
```
### 2. Usage
```
$ ./sample_inference <model_path(.pb)> <input_node> <output_node> <batch_size> <input_height> <input_width> <input_channels>
```

## CNN examples
### ResNet50
#### Execute the following commands to run inference for resnet50 model:
##### Download the pretrained model ```resnet50_fp32_pretrained_model.pb```
```bash
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_fp32_pretrained_model.pb
```
##### Run the model
```bash
./sample_inference <model_path(to resnet50_fp32_pretrained_model.pb)> input predict 1280 224 224 3
```
##### Output
On successful execution, the output would be as follows.
```
Example Output data: Tensor<type: float shape: [1280,1001] values: [0.000189354629 0.000291871896 0.0012335889...]...>

Time taken: 2030.000000
FPS for 1280 images: 630.541870
```
>Note: The execution time/FPS listed in output is simply an example.

### InceptionV3
#### Execute the following commands to run inference for inceptionv3 model:
##### Download the pretrained model ```inceptionv3_fp32_pretrained_model.pb```
```bash
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv3_fp32_pretrained_model.pb
```
##### Run the model
```bash
./sample_inference <model_path(to inceptionv3_fp32_pretrained_model.pb)> input predict 1280 299 299 3
```
##### Output
On successful execution, the output would be as follows.
```
Example Output data: Tensor<type: float shape: [1280,1001] values: [8.6072534e-05 0.000105876556 0.000373733055...]...>

Time taken: 2408.000000
FPS for 1280 images: 531.561462
```
>Note: The execution time/FPS listed in output is simply an example.
