# zentf C++ Examples
This document provides examples for running inference on CNN models using zentf C++ interface.

## Table of Contents
- [Set-up zentf for C++ user interface](#set_up-zentf-for-c_user-interface)
- [Build C++ inference application](#build-c_inference-application)
- [CNN examples](#cnn-examples)

## Set-up zentf for C++ user interface

Please follow the instructions in the [README.md](https://github.com/amd/ZenDNN-tensorflow-plugin/blob/main/scripts/c%2B%2B/README.md) file for setting up zentf.

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
2025-03-20 08:34:30.781698: I tensorflow/core/util/port.cc:180] ZenDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ZENDNN_OPTS=0`.
2025-03-20 08:34:30.782170: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-20 08:34:30.784504: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-20 08:34:30.789888: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1742459670.798156 1223555 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1742459670.800619 1223555 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-03-20 08:34:30.821778: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-03-20 08:34:30.824145: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
2025-03-20 08:34:30.883601: E tensorflow/core/framework/op_kernel.cc:1772] OpKernel ('op: "_ZenQuantizedAvgPool" device_type: "CPU" constraint { name: "T" allowed_values { list { type: DT_QINT8 } } }') for unknown op: _ZenQuantizedAvgPool
2025-03-20 08:34:30.883644: E tensorflow/core/framework/op_kernel.cc:1772] OpKernel ('op: "_ZenQuantizedAvgPool" device_type: "CPU" constraint { name: "T" allowed_values { list { type: DT_QUINT8 } } }') for unknown op: _ZenQuantizedAvgPool
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1742459670.917564 1223555 mlir_graph_optimization_pass.cc:401] MLIR V1 optimization pass is not enabled

Random input data: Tensor<type: float shape: [1280,224,224,3] values: [[[0.840187728 0.394382924 0.783099234]]]...>
2025-03-20 08:34:34.091997: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type CPU is enabled.
OMP: Warning #181: OMP_PROC_BIND: ignored because GOMP_CPU_AFFINITY has been defined
Start  Thu Mar 20 08:34:56 2025
Stop Thu Mar 20 08:38:19 2025


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
2025-03-20 08:43:07.473306: I tensorflow/core/util/port.cc:180] ZenDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ZENDNN_OPTS=0`.
2025-03-20 08:43:07.473779: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-20 08:43:07.476079: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-20 08:43:07.481582: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1742460187.490377 1225375 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1742460187.492839 1225375 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-03-20 08:43:07.514394: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-03-20 08:43:07.516707: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
2025-03-20 08:43:07.584674: E tensorflow/core/framework/op_kernel.cc:1772] OpKernel ('op: "_ZenQuantizedAvgPool" device_type: "CPU" constraint { name: "T" allowed_values { list { type: DT_QINT8 } } }') for unknown op: _ZenQuantizedAvgPool
2025-03-20 08:43:07.584711: E tensorflow/core/framework/op_kernel.cc:1772] OpKernel ('op: "_ZenQuantizedAvgPool" device_type: "CPU" constraint { name: "T" allowed_values { list { type: DT_QUINT8 } } }') for unknown op: _ZenQuantizedAvgPool
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1742460187.623462 1225375 mlir_graph_optimization_pass.cc:401] MLIR V1 optimization pass is not enabled

Random input data: Tensor<type: float shape: [1280,299,299,3] values: [[[0.840187728 0.394382924 0.783099234]]]...>
2025-03-20 08:43:13.274976: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type CPU is enabled.
OMP: Warning #181: OMP_PROC_BIND: ignored because GOMP_CPU_AFFINITY has been defined
Start  Thu Mar 20 08:43:39 2025
Stop Thu Mar 20 08:47:39 2025


Example Output data: Tensor<type: float shape: [1280,1001] values: [8.6072534e-05 0.000105876556 0.000373733055...]...>

Time taken: 2408.000000
FPS for 1280 images: 531.561462
```
>Note: The execution time/FPS listed in output is simply an example.
