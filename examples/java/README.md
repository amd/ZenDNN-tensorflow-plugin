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
cd <Path to zentf plugin parent folder>/ZenDNN_TensorFlow_Plugin/examples/java
mvn clean package
```

#### Run the model.
```bash
java -cp target/tensorflow-benchmark-0.1-jar-with-dependencies.jar org.tensorflow.benchmark.RunWideDeeplarge <path to wide deep large .pb model> <batch size>
```

#### Output
On successful execution, the output would be as follows.
```
Batch Size = <batch size>
Output:
0
End of execution.
```
