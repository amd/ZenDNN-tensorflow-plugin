# zentf Python Examples

This document provides examples for running inference on various models using the ZenDNN TensorFlow plugin with Python user interface. Note that you may need to install additional packages in your environment if not already present. Assuming you have installed ZenDNN TensorFlow plugin in your environment, you can install the rest of the packages by running:

```bash
pip install -r requirements.txt
```

## Table of Contents
- [BERT](#bert)
- [OPT](#opt)
- [ResNet](#resnet)
- [Wide&Deep](#widedeep)

Please refer to the [README.md](../../README.md) for installation of zentf and setting the recommended environment settings.

## BERT

Execute the following command to run inference for bert model:

```bash
python bert_inference.py
```

#### Output
On successful execution, the output would be as follows.
```bash
Last hidden states shape:(1, 8, 1024)
```

## OPT

Execute the following command to run inference for OPT model:

```bash
python opt_inference.py
```

#### Output

On successful execution, the output would be as follows.

```
Are you conscious? Can you talk?
I can talk, but I can't really think
```

## ResNet

Execute the following command to run inference for ResNet model:

```bash
python resnet_inference.py
```

#### Output

On successful execution, the output would be as follows.
```
plane, carpenter's plane, woodworking plane
```

## Wide&Deep
For running this model, ensure that you download the following files in your directory:

#### Download and preprocess the dataset

Run the following command to download eval.csv:
```bash
wget -nc https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv -O eval.csv
```

Download the dataset preprocessing script 'preprocess_csv_tfrecords.py' from: https://github.com/intel/ai-reference-models/blob/main/datasets/large_kaggle_advertising_challenge/preprocess_csv_tfrecords.py

Convert CSV to TFRecords

```bash
python preprocess_csv_tfrecords.py \
    --inputcsv-datafile eval.csv \
    --calibrationcsv-datafile train.csv \
    --outputfile-name preprocessed
```

#### Environment Variables Setup

```bash
TF_NUM_INTRAOP_THREADS=128
TF_NUM_INTEROP_THREADS=1
'TF_NUM_INTRAOP_THREADS' defines the number of threads ('128' used in this example) used for intra-op parallelism, optimizing operations like matrix multiplications and TF_NUM_INTEROP_THREADS controls the number of threads ('1' used in this example) used for inter-op parallelism, helping to execute independent operations concurrently.
```

Execute the following command to run inference for Wide&deep model:
```bash
python tfrecsys.py --datafile *.tfrecords
```

#### Output

On successful execution, the output would be as follows.
```
--------------------------------------------------
Output tensor shape: (512, 2)
Output tensor value: [[0.90496653 0.09503351]
 [0.86235213 0.13764785]
 [0.78273624 0.21726371]
 ...
 [0.7851486  0.21485138]
 [0.98395133 0.01604872]
 [0.558111   0.44188896]]
--------------------------------------------------
```
