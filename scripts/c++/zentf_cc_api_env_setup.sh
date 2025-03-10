#*******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Enable TensorFlow-ZenDNN Plug-in.
export TF_ENABLE_ZENDNN_OPTS=1
echo "TF_ENABLE_ZENDNN_OPTS=$TF_ENABLE_ZENDNN_OPTS"
export TF_ENABLE_ONEDNN_OPTS=0
echo "TF_ENABLE_ONEDNN_OPTS=$TF_ENABLE_ONEDNN_OPTS"

# Setting ZenDNN Environment variables.
echo "Setting ZenDNN Environment variables."
echo "The following settings gave us the best results. However, these details should be verified by the user empirically."
# Switch to set zen mempool optimization.
# Default value is 1, but for this release we recommend :
#   2 - For NLP and LLM models.
#   3 - For CNN models.
export ZENDNN_ENABLE_MEMPOOL=2
echo "ZENDNN_ENABLE_MEMPOOL=$ZENDNN_ENABLE_MEMPOOL"
# Variable to set the max no. of tensors that can be used inside zen memory pool.
export ZENDNN_TENSOR_POOL_LIMIT=1024
echo "ZENDNN_TENSOR_POOL_LIMIT=$ZENDNN_TENSOR_POOL_LIMIT"
# Enable/Disable fixed max size allocation for Persistent tensor with
# zen memory pool optimization. By default, its disabled.
export ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=0
echo "ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=$ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE"
# Convolution algo path.
#   0. AUTO
#   1. GEMM
#   2. WINOGRAD
#   3. NHWC BLOCKED
# If env variable is not set, GEMM will be selected as default path.
# We recommend NHWC BLOCKED format for the best possible performance.
export ZENDNN_CONV_ALGO=3
echo "ZENDNN_CONV_ALGO=$ZENDNN_CONV_ALGO"

# Matmul Algorithms Settings. By default, it is ZENDNN_MATMUL_ALGO=FP32:4,BF16:4.
echo "By default, ZENDNN_MATMUL_ALGO=FP32:4,BF16:4"
# We recommend to override the default settings for NLPs & LLMs models by 
# uncommenting the following 'export' and 'echo' commands.
# Note: Do not uncomment for AMP (Auto-Mixed Precision) mode runs of any models.
# export ZENDNN_MATMUL_ALGO=FP32:2,BF16:0
# echo "Overriding with ZENDNN_MATMUL_ALGO=$ZENDNN_MATMUL_ALGO"

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

# OMP settings
# The below settings are for Turin. Please update it based on your machine.
echo -e "\nBy default, OMP settings are based on Turin machine."
echo "Please update it based on your machine."
export OMP_NUM_THREADS=128
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
export GOMP_CPU_AFFINITY="0-127"
echo "GOMP_CPU_AFFINITY=$GOMP_CPU_AFFINITY"
export OMP_PROC_BIND=FALSE
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
export OMP_WAIT_POLICY=ACTIVE
echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
export OMP_DYNAMIC=FALSE
echo "OMP_DYNAMIC=$OMP_DYNAMIC"

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
