#*******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#*******************************************************************************

#!/bin/bash

#-----------------------------------------------------------------------------
#   zendnn_tf_cc_api_env_setup.sh
#   Usage: This script needs to run first to setup environment variables
#          before using the tf-zendnn-plugin c++ api library.
#
#   This script does following:
#   -Enable TensorFlow-ZenDNN Plugin.
#   -Sets the ZenDNN Environment variables.
#   -Exports LIBRARY_PATH and LD_LIBRARY_PATH.
#   -Creates necessary soft links.
#----------------------------------------------------------------------------

# Enable TensorFlow-ZenDNN Plugin.
export TF_ENABLE_ZENDNN_OPTS=1
echo "TF_ENABLE_ZENDNN_OPTS=$TF_ENABLE_ZENDNN_OPTS"
export TF_ENABLE_ONEDNN_OPTS=0
echo "TF_ENABLE_ZENDNN_OPTS=$TF_ENABLE_ONEDNN_OPTS"

# Setting ZenDNN Environment variables.
echo "Setting ZenDNN Environment variables."
echo "The following settings gave us the best results. However, these details should be verified by the user empirically."
# Switch to set zen mempool optimization.
# By default, zen mempool is enabled with option '2' (node based mempool).
export ZENDNN_ENABLE_MEMPOOL=2
echo "ZENDNN_ENABLE_MEMPOOL=$ZENDNN_ENABLE_MEMPOOL"
# Variable to set the max no. of tensors that can be used inside zen memory pool.
export ZENDNN_TENSOR_POOL_LIMIT=1024
echo "ZENDNN_TENSOR_POOL_LIMIT=$ZENDNN_TENSOR_POOL_LIMIT"
# Enable/Disable fixed max size allocation for Persistent tensor with
# zen memory pool optimization. By default, its disabled.
export ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=0
echo "ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=$ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE"
# Switch to set Convolution algo type.
# By default, algo type is set to '4' which is Direct convolution with
# only filters in blocked memory format.
export ZENDNN_CONV_ALGO=4
echo "ZENDNN_CONV_ALGO=$ZENDNN_CONV_ALGO"
# Switch to set Matmul algo type.
# By default, its set to '4' for FP32 which is BRGEMM kernel path.
export ZENDNN_MATMUL_ALGO=FP32:4
echo "ZENDNN_MATMUL_ALGO=$ZENDNN_MATMUL_ALGO"
# Enable/Disable primitive reuse.
export TF_ZEN_PRIMITIVE_REUSE_DISABLE=FALSE
echo "TF_ZEN_PRIMITIVE_REUSE_DISABLE=$TF_ZEN_PRIMITIVE_REUSE_DISABLE"
# Set primitive caching capacity.
export ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY=$ZENDNN_PRIMITIVE_CACHE_CAPACITY"
# Variable to control ZenDNN logs.
# By default, ZenDNN logs are disabled.
export ZENDNN_LOG_OPTS=ALL:0
echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"

export TF_CC_API_ZENDNN_ROOT=$(pwd)
if [ -z "$TF_CC_API_ZENDNN_ROOT" ];
then
    echo "Error: Environment variable TF_CC_API_ZENDNN_ROOT needs to be set"
    return
else
    echo "TF_CC_API_ZENDNN_ROOT=$TF_CC_API_ZENDNN_ROOT"
fi

# Exports necessary library paths for plugin.
export LIBRARY_PATH=$TF_CC_API_ZENDNN_ROOT/lib-tensorflow-plugins/:$LIBRARY_PATH
if [ -z "$LIBRARY_PATH" ];
then
    echo "Error: Environment variable LIBRARY_PATH needs to be set"
    return
else
    echo "LIBRARY_PATH=$LIBRARY_PATH"
fi

export LD_LIBRARY_PATH=$TF_CC_API_ZENDNN_ROOT/lib-tensorflow-plugins/:$LD_LIBRARY_PATH
if [ -z "$LD_LIBRARY_PATH" ];
then
    echo "Error: Environment variable LD_LIBRARY_PATH needs to be set"
    return
else
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi

# TODO(plugin) : find a better solution to create softlink for libomp.so.5.
cd $TF_CC_API_ZENDNN_ROOT/lib-tensorflow-plugins
ln -sf libiomp5.so libomp.so.5

cd $TF_CC_API_ZENDNN_ROOT
