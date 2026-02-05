#*******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#
#   This script does following:
#   -Enables TensorFlow-ZenDNN Plugin.
#   -Sets the ZenDNN environment variables.
#
#----------------------------------------------------------------------------
# Set the user interface Python/C++/Java. The first argument of this script
# specifies the interface. If the first argument is not specified then the
# default environment settings will work for Python/C++.

interface="$1"

# Enable TensorFlow-ZenDNN Plug-in.
export TF_ENABLE_ZENDNN_OPTS=1
echo "TF_ENABLE_ZENDNN_OPTS=$TF_ENABLE_ZENDNN_OPTS"
export TF_ENABLE_ONEDNN_OPTS=0
echo "TF_ENABLE_ONEDNN_OPTS=$TF_ENABLE_ONEDNN_OPTS"

# Setting ZenDNN Environment variables.
echo "Setting ZenDNN Environment variables."
echo "The following settings gave us the best results. However, these details should be verified by the user empirically."

if [ "$interface" = "java" ]; then
    export KMP_BLOCKTIME=1
    export KMP_TPAUSE=0
    export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
    export KMP_PLAIN_BARRIER_PATTERN=dist,dist
    export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
    export KMP_AFFINITY=granularity=fine,compact,1,0
    echo "KMP_BLOCKTIME=$KMP_BLOCKTIME"
    echo "KMP_TPAUSE=$KMP_TPAUSE"
    echo "KMP_FORKJOIN_BARRIER_PATTERN=$KMP_FORKJOIN_BARRIER_PATTERN"
    echo "KMP_PLAIN_BARRIER_PATTERN=$KMP_PLAIN_BARRIER_PATTERN"
    echo "KMP_REDUCTION_BARRIER_PATTERN=$KMP_REDUCTION_BARRIER_PATTERN"
    echo "KMP_AFFINITY=$KMP_AFFINITY"
fi

# TODO(plugin): Add the ZenDNNL environment variables.
# ZENDNNL_MATMUL_ALGO options:
# -1: none (default selection logic)
# 0: dynamic_dispatch (LOA only)
# 1: aocl_dlp_blocked
# 2: onednn_blocked
# 3: libxsmm_blocked (LOA only)
# 4: aocl_dlp
# 5: onednn
# 6: libxsmm (LOA only)
# auto: auto_tuner (LOA only)
#
# For FP32 and direct BF16 models: use ZENDNNL_MATMUL_ALGO=1
# For AMP models (when TF_ZENDNN_PLUGIN_BF16=1): strictly use ZENDNNL_MATMUL_ALGO=4 or 5 only
if [ "$TF_ZENDNN_PLUGIN_BF16" = "1" ]; then
    # AMP models: use 4 as per recommendations
    export ZENDNNL_MATMUL_ALGO=4
else
    # FP32 and direct BF16 models: use 1 as default
    export ZENDNNL_MATMUL_ALGO=1
fi
echo "ZENDNNL_MATMUL_ALGO=$ZENDNNL_MATMUL_ALGO"
export USE_ZENDNN_MATMUL_DIRECT=1
echo "USE_ZENDNN_MATMUL_DIRECT=$USE_ZENDNN_MATMUL_DIRECT"

# OMP settings
# The below settings are for Turin. Please update it based on your machine.
echo -e "\nBy default, OMP settings are based on Turin machine."
echo "Please update it based on your machine or usecase."
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
