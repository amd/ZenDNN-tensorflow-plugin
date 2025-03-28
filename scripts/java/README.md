# Use zentf plugin with Java API of TensorFlow

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Set-up zentf plugin for TensorFlow-Java](#set-up-zentf-plugin-for-tensorflow-java)
- [Examples](#examples)

## Introduction
This file detail on setting up zentf plugin for [TensorFlow-Java](https://github.com/tensorflow/java). This set-up enables Java applications to exploit zentf in DNN inference using TensorFlow.

## Prerequisites
Before building the project, ensure you have the following installed:
- Maven 3.6 or higher
- Java Development Kit (JDK) 11 or higher
- Environment variable JAVA_HOME set to your JDK installation path
- GLIBC v2.33 or higher

Note that you may need to set JAVA_HOME to the appropriate path for buidling this project with Maven. For example, JAVA_HOME must be set to the path `/usr/lib/jvm/java-<java-version>-openjdk-amd64`.

For example:-
```bash
For Ubuntu OS:-
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

For RHEL OS:-
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
export PATH=$JAVA_HOME/bin:$PATH
```

Note: Above we are assuming that the user has Java-v11 installed.

It could be possible that java version might be different in your machine, so please check what version is present at `/usr/lib/jvm` location and set it accordingly.

## Set-up zentf plugin for TensorFlow-Java
As of now (i.e. v1.0.0 of TensorFlow-Java), there is no support for plugins. So, TensorFlow-Java builds available publicly cannot be used.

The following steps are required to enable zentf plugin for TensorFlow-Java,
- Build TensorFlow-Java from source with changes for plugin
- Set-up zentf plugin

Note: TensorFlow-Java v1.0.0 supports TensorFlow v2.16.2.

### Build TensorFlow-Java from source with changes for plugin

Run the script 'build_tf_java.sh' to get the source code of TensorFlow-Java, apply the source code changes to support plugin, build and install TensorFlow-Java from source.
```bash
cd <Path to zentf plugin parent folder>/ZenDNN_TensorFlow_Plugin/scripts/java
bash build_tf_java.sh
```
### Set-up zentf plugin
Download zentf C++ plugin package of 5.0.2 release from [AMD Developer Forum](https://www.amd.com/en/developer/zendnn.html).

Set the environment variable 'LD_LIBRARY_PATH' with the path to zentf plugin libraries.
```bash
unzip ZENTF_v5.0.2_C++_API.zip
export LD_LIBRARY_PATH=<Path to zentf C++ parent folder>/ZENTF_v5.0.2_C++_API/lib-tensorflow-plugins
```

Set ZenDNN specific environment variables as shown below,
```bash
cd <Path to zentf C++ parent folder>/ZENTF_v5.0.2_C++_API
source zentf_env_setup.sh java
```

## Examples
To try the set-up made on an example inference application, please refer README file from './examples/java' folder of this repository.
