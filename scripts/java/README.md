# Use zentf plugin with Java API of TensorFlow

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Build TensorFlow-Java](#build-tensorflow-java)
- [Set-up zentf plugin for TensorFlow-Java](#set-up-zentf-plugin-for-tensorflow-java)
- [Examples](#examples)

## Introduction
This file detail on setting up zentf plugin for [TensorFlow-Java](https://github.com/tensorflow/java). This set-up enables Java applications to exploit zentf in DNN inference using TensorFlow.

## Prerequisites
Before building the project, ensure you have the following installed:
- Git
- Bazel 7.4.1
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

You can install Bazel 7.4.1 by following the steps below:
```bash
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel-7.4.1
```

## Build TensorFlow-Java
**Note:** As of now, zentf 5.2.0 supports only the `tf-2.20` branch of TensorFlow-Java, which must be built from source.

A build script `build_tf_java.sh` is provided to automate this build process.

The script performs the following steps:
1. Clones the [TensorFlow-Java](https://github.com/tensorflow/java) repository.
2. Verifies that all prerequisites (Bazel, Maven, Java) are installed.
3. Checks out the `tf-2.20` branch.
4. Builds TensorFlow-Java using Maven.

To run the build script:
```bash
source build_tf_java.sh
```

Note: The script uses `source` (instead of `bash`) as it uses `return` statements for error handling. If the directory `tf_java.git` already exists, the script will prompt you to remove it before proceeding.

## Set-up zentf plugin for TensorFlow-Java
Plugin support is now available in TensorFlow Java as of version 1.2.0-SNAPSHOT.

Note: TensorFlow-Java v1.2.0-SNAPSHOT supports TensorFlow v2.20.0.

### Set-up zentf plugin
Download zentf C++ plugin package of 5.2.0 release from [AMD Developer Forum](https://www.amd.com/en/developer/zendnn.html).

Set the environment variable 'LD_LIBRARY_PATH' with the path to zentf plugin libraries.
```bash
unzip ZENTF_v5.2.0_C++_API.zip
export LD_LIBRARY_PATH=<Path to zentf C++ parent folder>/ZENTF_v5.2.0_C++_API/lib-tensorflow-plugins
```

Set ZenDNN specific environment variables as shown below,
```bash
cd <Path to zentf C++ parent folder>/ZENTF_v5.2.0_C++_API
source zentf_env_setup.sh
```

Load zentf plugin native library in the application code.
```java
// Load the TF_LoadPluggableDeviceLibrary package
import static org.tensorflow.internal.c_api.global.tensorflow.TF_LoadPluggableDeviceLibrary;
import org.tensorflow.internal.c_api.TF_Library;
```
```java
  public static void main(String[] params) {
    // The libamdcpu_plugin_cc.so file should be present in LD_LIBRARY_PATH.
    String zentf_path = "libamdcpu_plugin_cc.so";
    load_zentf(zentf_path);
    System.out.println("Using zendnn.");

    // Rest of the code
}

private static void load_zentf(String filename) {
    TF_Status status = TF_Status.newStatus();
    TF_Library h = TF_LoadPluggableDeviceLibrary(filename, status);
    status.throwExceptionIfNotOK();
}
```

## Examples
To try the set-up made on an example inference application, please refer [zentf Java Examples](../../examples/java/README.md).
