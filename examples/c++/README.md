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
cd <cpp_package_name>/
g++ examples/sample_inference.cpp -o sample_inference -I./<tf_folder>/tensorflow/include -L./<tf_folder>/tensorflow/ -ltensorflow_framework -ltensorflow_cc -Wl,-rpath=./<tf_folder>/tensorflow/ -std=c++17
```
### 2. Usage
```
./sample_inference <model_path(.pb)> <input_node> <output_node> <batch_size> <input_height> <input_width> <input_channels>
```

## CNN examples
### ResNet50
#### Execute the following commands to run inference for resnet50 model:
##### Download the pretrained model ```resnet50_v1.pb```
```bash
wget https://zenodo.org/records/2535873/files/resnet50_v1.pb
```
##### Run the model
```bash
./sample_inference <model_path(to resnet50_v1.pb)> input_tensor softmax_tensor 1280 224 224 3
```
##### Output
On successful execution, the output would be as follows.
```
Example Output data: Tensor<type: float shape: [1280,1001] values: [0.000107617641 0.000108799642 0.000492911378...]...>

Time taken: 11965.825195
FPS for 1280 images: 106.971306
```
>Note: The execution time/FPS listed in output is simply an example.

### MobilenetV1
#### Execute the following commands to run inference for mobilenetv1 model:
##### Download the pretrained model ```mobilenet_v1_1.0_224_frozen.pb```
```bash
wget https://web.archive.org/web/2id_/https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/mobilenet_v1_1.0_224_frozen.pb
```
##### Run the model
```bash
./sample_inference <model_path(to mobilenet_v1_1.0_224_frozen.pb)> input MobilenetV1/Predictions/Reshape_1 1280 224 224 3
```
##### Output
On successful execution, the output would be as follows.
```
Example Output data: Tensor<type: float shape: [1280,1001] values: [2.90372441e-06 7.68667724e-06 3.70206799e-05...]...>

Time taken: 904.982788
FPS for 1280 images: 1414.391479
```
>Note: The execution time/FPS listed in output is simply an example.
