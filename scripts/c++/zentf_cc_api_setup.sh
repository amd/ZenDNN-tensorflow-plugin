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
#   This script does following:
#   -Exports LIBRARY_PATH and LD_LIBRARY_PATH.
#   -Creates necessary soft links.
#----------------------------------------------------------------------------

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
