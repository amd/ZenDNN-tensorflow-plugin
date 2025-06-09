#*******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# This script automates the build process of TF-Java with zentf Plugin support.

# Please refer the README.md file on how to use this script.

current_dir="$PWD"
echo "current_dir: $current_dir"

# Clone Java repo from GitHub.
if [ -d tf_java.git ]; then
    read -p "Directory 'tf_java.git' exists. Do you want to remove it? (y/n): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf tf_java.git
    else
        echo "Keeping existing 'tf_java.git' directory."
        return 1
    fi
fi
git clone https://github.com/tensorflow/java.git tf_java.git
wget https://patch-diff.githubusercontent.com/raw/tensorflow/java/pull/605.patch

# Change directory to java.
cd $current_dir/tf_java.git
# Checking out to v1.0.0 relase tag.
git checkout v1.0.0

# Check if bazel is installed.
if ! command -v bazel &> /dev/null; then
    echo "Error: bazel is not installed"
    # You can install bazel by following below steps.
    # sudo apt install apt-transport-https curl gnupg -y
    # curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
    # sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
    # echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    # sudo apt update && sudo apt install bazel-6.5.0
    return 1
else
    echo "bazel version: $(bazel version)"
fi
echo ""

# Check if maven is installed.
if ! command -v mvn &> /dev/null; then
    echo "Error: maven is not installed"
    return 1
else
    echo "maven version: $(mvn -version)"
fi
echo ""

# Check if java is installed.
if ! command -v java &> /dev/null; then
    echo "Error: java is not installed"
    return 1
else
    echo "java version: $(java -version 2>&1)"
fi
echo ""

# Apply the patch.
git apply ../605.patch

# Build TensorFlow Java.
if ! mvn clean install; then
    echo "Error: Maven build failed"
    return 1
else
    echo "Maven build completed successfully"
fi

cd $current_dir