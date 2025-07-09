# Use TensorFlow-ZenDNN Plug-in with C++ APIs

This file detail on setting up zentf for C++ interface. This set-up enables C++ applications to exploit zentf in DNN inference using TensorFlow.

You can download zentf C++ package from [AMD Developer Forum](https://www.amd.com/en/developer/zendnn.html).

## C++ package structure
```
.
├── examples/
├── lib-tensorflow-plugins/
├── README.md
├── tensorflow_<version>/
├── zentf_cc_api_setup.sh
└── zentf_env_setup.sh
```

## How to use C++ package?

### 1. Unzip the package
```
$ unzip <cpp_package_name>.zip
$ cd <cpp_package_name>/
```
### 2. Set required library path
```
$ source zentf_cc_api_setup.sh
```
### 3. Set ZenDNN specific environment variables
```
$ source zentf_env_setup.sh
```
Set up is done!

## Examples
To try the set-up made on an example inference application, please refer [zentf C++ Examples](../../examples/c++/README.md).
