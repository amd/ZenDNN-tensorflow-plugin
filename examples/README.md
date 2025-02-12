# Examples
Given below are example commands for running inference on various models using TensorFlow. Please note that additional dependencies may be required based on your environment.

Before proceeding, ensure that the zentf plugin is installed. You can download it [here](https://github.com/amd/ZenDNN-tensorflow-plugin/tree/r5.0.1#installation-guide).

Once the zentf plugin is set up, you can install the remaining dependencies by running:
```bash
pip install -r requirements.txt
```

Please refer the script  [zentf_env_setup.sh](https://github.com/amd/ZenDNN-tensorflow-plugin/blob/main/scripts/zentf_env_setup.sh)  for recommended environment settings for zentf.

The script shall be updated in reference to the "Performance Tuning" section of ZenDNN User Guide and used.
```bash
source ../scripts/zentf_env_setup.sh

```

## BERT
### Execute the following command to run inference for bert model:
```bash
python bert_inference.py
```
### Output
Last hidden states shape:(1, 8, 1024)

## OPT
### Execute the following command to run inference for opt model:
```bash
python opt_inference.py
```
### Output
Are you conscious? Can you talk?
I can talk, but I can't really think

## ResNet
### Execute the following command to run inference for resnet model:
```bash
python resnet_inference.py
```
### Output
plane, carpenter's plane, woodworking plane

## Wide&Deep
For running this model, ensure that you download the following files in your directory:

### Download Dataset- Run the following command to download eval.csv:
```bash
 wget -nc https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv -O $DATASET_DIR/eval.csv
```

### Download preprocess_csv_tfrecords.py from the link below:
https://github.com/intel/ai-reference-models/blob/main/datasets/large_kaggle_advertising_challenge/preprocess_csv_tfrecords.py

### *.tfrecords : Run the following to convert CSV to TFRecords
```bash
if [ ! -f $DATASET_DIR/eval_preprocessed.tfrecords ] ; then
    cd $DATASET_DIR
    python preprocess_csv_tfrecords.py \
        --inputcsv-datafile eval.csv \
        --calibrationcsv-datafile train.csv \
        --outputfile-name preprocessed
fi
```

### Please set following environment variables as below
TF_NUM_INTRAOP_THREADS=<No. of cores>

TF_NUM_INTEROP_THREADS=1

TF_NUM_INTRAOP_THREADS: Defines the number of threads used for intra-op parallelism, optimizing operations like matrix multiplications.
TF_NUM_INTEROP_THREADS: Controls the number of threads used for inter-op parallelism, helping to execute independent operations concurrently.

### Execute the following command to run inference using tfrecsys.py:
```bash
python tfrecsys.py --datafile *.tfrecords
```
