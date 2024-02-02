# Use TensorFlow-ZenDNN Plug-in with C++ APIs

This file describes how to use C++ package and inference the Public CNN models with it.

## How to use C++ package?

### 1. Unzip the package
```
$ unzip <cpp_package_name>.zip
$ cd <cpp_package_name>/
```
### 2. Set env and library path
```
$ source zendnn_tf_cc_api_env_setup.sh
```
Set up is done!

## Optimize the model for inference

### Optimize the model
```
$ cd <cpp_package_name>/
$ ./<tf_folder>/tensorflow/tools/transform_graph
    --in_graph=<model_path(.pb)>
    --out_graph=<optimized_model_name(.pb)>
    --inputs=<input_node>
    --outputs=<output_node>
    --transforms='strip_unused_nodes(type=float, shape=<input_shape>) remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'
```

## Sample inferencing of the model

### 1. Compile the sample inference script
```
$ g++ sample_inference.cpp -o sample_inference -I./<tf_folder>/tensorflow/include -L./<tf_folder>/tensorflow/ -ltensorflow_framework -ltensorflow_cc -Wl,-rpath=./<tf_folder>/tensorflow/ -std=c++17
```
### 2. Run the model
```
$ ./sample_inference <model_path(.pb)> <input_node> <output_node> <batch_size> <input_height> <input_width> <input_channels>
```
