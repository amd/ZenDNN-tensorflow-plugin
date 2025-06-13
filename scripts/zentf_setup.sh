#*******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#   -Tensorflow v2.19 has to be installed.
#
#   This script does the following:
#   -Configures the Build options for TensorFlow-ZenDNN Plug-in.
#   -Generates a wheel file.
#   -Installs whl package.
#   -Enables TensorFlow-ZenDNN Plug-in.
#   -Sets ZenDNN Environment variables.
#----------------------------------------------------------------------------

# Configuring Build options.
echo "Configure TensorFlow-ZenDNN Plug-in"
./configure

# Building the TensorFlow-ZenDNN Plug-in.
echo "Building the TensorFlow-ZenDNN Plug-in, please wait..."
bazel build  -c opt //tensorflow_plugin/tools/pip_package:build_pip_package --verbose_failures --copt=-Wall --copt=-Werror --spawn_strategy=standalone >& bazel_build_output.txt

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

# Extract the script location.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/zentf_env_setup.sh

# Execute sample kernel.
python tests/softmax.py

echo "Setup is done!"
