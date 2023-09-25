#*******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#*******************************************************************************

#!/bin/bash

#----------------------------------------------------------------------------
#   Prerequisites:
#   -An active conda environment.
#   -Tensorflow v2.14 has to be installed.
#
#   This script does the following:
#   -Configures the Build options for TensorFlow-ZenDNN Plug-in.
#   -Generates a wheel file.
#   -Installs whl package.
#   -Enables TensorFlow-ZenDNN Plug-in.
#   -Sets ZenDNN Environment variables.
#----------------------------------------------------------------------------

# Installing neccessary requirements.
pip install -r requirements.txt

# Configuring Build options.
echo "Configure TensorFlow-ZenDNN Plug-in"
./configure

# Building the TensorFlow-ZenDNN Plug-in.
echo "Building the TensorFlow-ZenDNN Plug-in, please wait..."
bazel build  -c opt --show_result 0 //tensorflow_plugin/tools/pip_package:build_pip_package --verbose_failures --spawn_strategy=standalone >& bazel_build_output.txt
if [[ $? -ne 0 ]]
then
    echo "Build failed. Please see the bazel_build_output.txt file for more details."
    return
else
    echo "Build Successful."
fi

# Generate wheel file.
echo "Generating wheel file."
bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package .

# Installing wheel file.
WHL_FILE=$(find *.whl -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)
echo "Installing wheel file."
python -m pip install $WHL_FILE
echo "Installation successful."

# Enable TensorFlow-ZenDNN Plug-in.
export TF_ENABLE_ZENDNN_OPTS=1
export TF_ENABLE_ONEDNN_OPTS=0

# Setting ZenDNN Environment variables.
echo "Setting ZenDNN Environment variables."
echo "The following settings gave us the best results. However, these details should be verified by the user empirically."
export ZENDNN_LOG_OPTS=ALL:0
echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
export TF_ZEN_PRIMITIVE_REUSE_DISABLE=False
echo "TF_ZEN_PRIMITIVE_REUSE_DISABLE=$TF_ZEN_PRIMITIVE_REUSE_DISABLE"
export ZENDNN_ENABLE_MEMPOOL=1
echo "ZENDNN_ENABLE_MEMPOOL=$ZENDNN_ENABLE_MEMPOOL"
export ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY=$ZENDNN_PRIMITIVE_CACHE_CAPACITY"
export ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=0
echo "ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=$ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE"
export ZENDNN_TENSOR_POOL_LIMIT=1024
echo "ZENDNN_TENSOR_POOL_LIMIT=$ZENDNN_TENSOR_POOL_LIMIT"
export ZENDNN_CONV_ALGO=4
echo "ZENDNN_CONV_ALGO=$ZENDNN_CONV_ALGO"
export ZENDNN_GEMM_ALGO=3
echo "ZENDNN_GEMM_ALGO=$ZENDNN_GEMM_ALGO"

# Execute sample kernel.
python tests/softmax.py

echo "Setup is done!"
