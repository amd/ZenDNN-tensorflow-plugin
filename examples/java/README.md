# zentf Java Examples

This document provides examples for running inference on various models using the ZenDNN TensorFlow plugin using Java interface.

## Table of Contents
- [Building TF For Java](#building-tf-for-java)
- [Run Wide Deep large model](#run-wide-deep-large-model)

## Building TF For Java

Please follow the instructions in the [README.md](../../scripts/java/README.md) file.

## Run Wide Deep large model

```bash
Download the model.
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.pb
```

```bash
cd <path to zentf plugin>/ZenDNN_TensorFlow_Plugin/examples/java
mvn clean package
```

#### Run the model.
```bash
java -cp target/tensorflow-benchmark-0.1-jar-with-dependencies.jar org.tensorflow.benchmark.RunWideDeeplarge <path to wide deep large .pb model> <batch size>
```

#### Output
On successful execution, the output would be as follows.
```
2025-03-18 23:27:27.549880: I tensorflow/core/util/port.cc:140] ZenDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ZENDNN_OPTS=0`.
Using zendnn.
2025-03-18 23:27:27.734398: E tensorflow/core/framework/op_kernel.cc:1770] OpKernel ('op: "_ZenQuantizedAvgPool" device_type: "CPU" constraint { name: "T" allowed_values { list { type: DT_QINT8 } } }') for unknown op: _ZenQuantizedAvgPool
2025-03-18 23:27:27.734440: E tensorflow/core/framework/op_kernel.cc:1770] OpKernel ('op: "_ZenQuantizedAvgPool" device_type: "CPU" constraint { name: "T" allowed_values { list { type: DT_QUINT8 } } }') for unknown op: _ZenQuantizedAvgPool
2025-03-18 23:27:27.881110: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-03-18 23:27:27.941346: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:388] MLIR V1 optimization pass is not enabled
2025-03-18 23:27:27.992795: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type CPU is enabled.
Batch Size = <batch size>
Output:
0
End of execution.
```
