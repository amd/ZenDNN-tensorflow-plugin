# TensorFlow-ZenDNN Plug-in For AMD CPUs
TensorFlow provides a [PluggableDevice mechanism (from v2.5)](https://blog.tensorflow.org/2021/06/pluggabledevice-device-plugins-for-TensorFlow.html) that enables modular, plug-and-play integration of device-specific code. AMD adopted PluggableDevice when implementing the TensorFlow-ZenDNN plug-in for it's EPYC<sup>TM</sup> CPUs ([See the blog announcement](https://blog.tensorflow.org/2023/03/enabling-optimal-inference-performance-on-amd-epyc-processors-with-the-zendnn-library.html)). TensorFlow-ZenDNN plug-in adds custom kernel implementations and operations specific to AMD EPYC<sup>TM</sup> CPUs to TensorFlow through its kernel and op registration C APIs.

TensorFlow-ZenDNN plug-in is a supplemental package to be installed alongside standard TensorFlow packages. From a TensorFlow developer’s perspective, the TensorFlow-ZenDNN plug-in approach simplifies the process of leveraging ZenDNN optimizations.

The following is a high-level block diagram for the TensorFlow-ZenDNN plug-in package which utilizes [ZenDNN](https://github.com/amd/ZenDNN) as the core inference library:

![TensorFlow-ZenDNN Plug-in](./images/zentf_overview.png)

This file shows how to implement, build, install and run a TensorFlow-ZenDNN plug-in for AMD CPUs.

## Supported OS
* Linux

## Prerequisites

| Tools/Frameworks | Version |
| :--------------: | :-----: |
| [Bazel](https://docs.bazel.build/versions/master/install-ubuntu.html) | >=3.1 |
| Git | >=1.8 |
| Python | >=3.9 and <=3.12 |
| [TensorFlow](https://www.tensorflow.org/) | ==2.16 |

# Installation Guide

>Note: Make sure you have active conda environment and TensorFlow v2.16 installed.

## Install via wheel file.
### 1. Download the wheel file from [here](http://ml-ci.amd.com:21096/view/ZenDNN/job/zendnn/job/tensorflow-zendnn-plugin-build-whl-release/).

### 2. Install wheel file using pip:
```
$ pip install zentf-4.2.0-cp310-cp310-linux_x86_64.whl
```

## Build and install from source.
### 1. Clone the repository
```
$ git clone https://github.com/amd/ZenDNN-tensorflow-plugin.git
$ cd ZenDNN-tensorflow-plugin/
```
### 2. Configuring &  Building the TensorFlow-ZenDNN Plug-in using script.
>Note: Configure & Build Tensorflow-ZenDNN Plug-in manually by following the steps [3-6].

```
The setup script will configure & build and install Tensorflow-ZenDNN Plug-in. It will also set the necessary environment variables of ZenDNN execution. However, these variables should be verified empirically.

ZenDNN-tensorflow-plugin$ source scripts/zentf_setup.sh
```
### 3. Configure the build options:
```
ZenDNN-tensorflow-plugin$ ./configure
You have bazel 5.3.0 installed.
Please specify the location of python. [Default is /home/user/anaconda3/envs/zentf-env/bin/python]:

Found possible Python library paths:
  /home/user/anaconda3/envs/zentf-env/lib/python3.10/site-packages
Please input the desired Python library path to use.  Default is [/home/user/anaconda3/envs/zentf-env/lib/python3.10/site-packages]

Do you wish to build TensorFlow plug-in with MPI support? [y/N]:
No MPI support will be enabled for TensorFlow plug-in.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]:

Configuration finished
```

### 4. Build the TensorFlow-ZenDNN Plug-in:
```
ZenDNN-tensorflow-plugin$ bazel build  -c opt //tensorflow_plugin/tools/pip_package:build_pip_package --verbose_failures --spawn_strategy=standalone
```

### 5. Generate python wheel file:
```
ZenDNN-tensorflow-plugin$ bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package .
  Note: It will generate and save python wheel file for TensorFlow-ZenDNN Plug-in into the current directory (i.e., ZenDNN-tensorflow-plugin/).
```

### 6. Install wheel file using pip:
```
ZenDNN-tensorflow-plugin$ pip install zentf-4.2.0-cp310-cp310-linux_x86_64.whl
```

**The build and installation from source is done!**

## Enable TensorFlow-ZenDNN Plug-in:
```
$ export TF_ENABLE_ZENDNN_OPTS=1
$ export TF_ENABLE_ONEDNN_OPTS=0
```
Note: To disable ZenDNN optimizations in your inference execution, you can set the corresponding ZenDNN environment variable `export TF_ENABLE_ZENDNN_OPTS=0`

## Execute sample kernel:
```
ZenDNN-tensorflow-plugin$ python tests/softmax.py
2024-03-27 22:51:57.292569: I tensorflow/core/util/port.cc:140] ZenDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ZENDNN_OPTS=0`.
2024-03-27 22:51:57.292832: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-03-27 22:51:57.295704: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-03-27 22:51:57.339363: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-27 22:51:57.969156: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Tensor("random_normal:0", shape=(10,), dtype=float32)
2024-03-27 22:51:58.407520: I tensorflow/core/common_runtime/direct_session.cc:380] Device mapping: no known devices.
2024-03-27 22:51:58.408159: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:388] MLIR V1 optimization pass is not enabled
random_normal/RandomStandardNormal: (RandomStandardNormal): /job:localhost/replica:0/task:0/device:CPU:0
2024-03-27 22:51:58.409983: I tensorflow/core/common_runtime/placer.cc:125] random_normal/RandomStandardNormal: (RandomStandardNormal): /job:localhost/replica:0/task:0/device:CPU:0
random_normal/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
2024-03-27 22:51:58.409998: I tensorflow/core/common_runtime/placer.cc:125] random_normal/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
random_normal: (AddV2): /job:localhost/replica:0/task:0/device:CPU:0
2024-03-27 22:51:58.410008: I tensorflow/core/common_runtime/placer.cc:125] random_normal: (AddV2): /job:localhost/replica:0/task:0/device:CPU:0
Softmax: (Softmax): /job:localhost/replica:0/task:0/device:CPU:0
2024-03-27 22:51:58.410018: I tensorflow/core/common_runtime/placer.cc:125] Softmax: (Softmax): /job:localhost/replica:0/task:0/device:CPU:0
random_normal/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2024-03-27 22:51:58.410025: I tensorflow/core/common_runtime/placer.cc:125] random_normal/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
random_normal/mean: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2024-03-27 22:51:58.410033: I tensorflow/core/common_runtime/placer.cc:125] random_normal/mean: (Const): /job:localhost/replica:0/task:0/device:CPU:0
random_normal/stddev: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2024-03-27 22:51:58.410041: I tensorflow/core/common_runtime/placer.cc:125] random_normal/stddev: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2024-03-27 22:51:58.429409: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type CPU is enabled.
[0.05660784 0.09040404 0.03201076 0.11204024 0.2344563  0.162052
 0.09466095 0.11205972 0.0752109  0.03049729]
```

# Resources
* [TensorFlow's Pluggable Device blog](https://blog.tensorflow.org/2021/06/pluggabledevice-device-plugins-for-TensorFlow.html)
* [AMD-TensorFlow blog](https://blog.tensorflow.org/2023/03/enabling-optimal-inference-performance-on-amd-epyc-processors-with-the-zendnn-library.html)
* [Download TensorFlow-ZenDNN Plug-in binary](http://ml-ci.amd.com:21096/view/ZenDNN/job/zendnn/job/tensorflow-zendnn-plugin-build-whl-release/)
* [TensorFlow-ZenDNN Plug-in User Guide](doc/TensorFlow-ZenDNN-Plugin-user-guide-v0.2.pdf)

# Performace tuning and Benchmarking
* Zentf v4.2.0 is supported with ZenDNN v4.2. Please see the section 2.7 of [ZenDNN User Guide](https://www.amd.com/content/dam/amd/en/documents/developer/zendnn-user-guide-4-1.pdf) for performance tuning guidelines.

# Support
* Please email zendnnsupport@amd.com for questions, issues, and feedback.
* Please submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/amd/ZenDNN-tensorflow-plugin/issues) page.

