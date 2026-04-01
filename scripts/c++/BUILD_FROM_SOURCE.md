# Building the ZenTF C++ Package from Source

This guide walks through building the TensorFlow-ZenDNN plug-in C++ package (`ZENTF_v5.2.1_C++_SOURCE_BUILD`) from source. The package enables ZenDNN-accelerated inference via the TensorFlow C++ API on AMD CPUs.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Any Linux distribution (tested on Ubuntu, RHEL) |
| **Conda** | Active conda environment |
| **TensorFlow** | v2.16.0 -- v2.21.0 installed in the environment |
| **Bazel** | Version matching the `.bazelversion` in the repo |
| **GCC** | >= 13.3 |
| **Python 3** | Available as `python3` |

---

## Directory Layout

ZenDNN_TensorFlow_Plugin repository must be cloned:

```
<workspace>/
└── ZenDNN_TensorFlow_Plugin/    # TensorFlow plug-in repo
```

---

### Step 1 -- Install TensorFlow

```bash
pip install tensorflow==2.21.0   # or any version in [2.16.0, 2.21.0]
```

Verify the installation:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

### Step 2 -- Configure the Build

```bash
cd <workspace>/ZenDNN_TensorFlow_Plugin
bazel clean --expunge --async
./configure
```

The `configure` script auto-detects your Python environment and TensorFlow installation. It writes `TF_HEADER_DIR` and `TF_SHARED_LIBRARY_DIR` into the Bazel configuration, pointing at the pip-installed TensorFlow headers and shared libraries.

### Step 3 -- Build the Plug-in Shared Library

```bash
bazel build -c opt //tensorflow_plugin:libamdcpu_plugin_cc.so \
    --verbose_failures \
    --copt=-Wall \
    --copt=-Werror \
    --spawn_strategy=standalone
```

This produces the plug-in shared library at:

```
bazel-bin/tensorflow_plugin/libamdcpu_plugin_cc.so
```

### Step 4 -- Assemble the C++ Package

Create the output directory and copy the build artifacts:

```bash
cd <workspace>
mkdir -p ZENTF_v5.2.1_C++_SOURCE_BUILD/lib-tensorflow-plugins

# Plug-in library
cp ZenDNN_TensorFlow_Plugin/bazel-bin/tensorflow_plugin/libamdcpu_plugin_cc.so \
   ZENTF_v5.2.1_C++_SOURCE_BUILD/lib-tensorflow-plugins/

# OpenMP runtime
cp ZenDNN_TensorFlow_Plugin/bazel-bin/tensorflow_plugin/libamdcpu_plugin_cc.so.runfiles/llvm_openmp/libiomp5.so \
   ZENTF_v5.2.1_C++_SOURCE_BUILD/lib-tensorflow-plugins/

# Helper scripts and sample code
cp ZenDNN_TensorFlow_Plugin/scripts/c++/zentf_cc_api_setup.sh  ZENTF_v5.2.1_C++_SOURCE_BUILD/
cp ZenDNN_TensorFlow_Plugin/scripts/zentf_env_setup.sh         ZENTF_v5.2.1_C++_SOURCE_BUILD/
cp ZenDNN_TensorFlow_Plugin/examples/c++/sample_inference.cpp   ZENTF_v5.2.1_C++_SOURCE_BUILD/
```

### Step 5 -- Download and Extract TensorFlow Headers & Libraries

The C++ API requires TensorFlow headers and shared libraries. Download the matching wheel and extract them:

```bash
TF_VERSION="2.21.0"    # must match the installed version

mkdir -p tf_wheel_tmp
pip download "tensorflow==${TF_VERSION}" --no-deps --only-binary=:all: -d tf_wheel_tmp
unzip tf_wheel_tmp/tensorflow*.whl -d "ZENTF_v5.2.1_C++_SOURCE_BUILD/tensorflow_${TF_VERSION}"
rm -rf tf_wheel_tmp
```

Create the required symlinks:

```bash
cd "ZENTF_v5.2.1_C++_SOURCE_BUILD/tensorflow_${TF_VERSION}/tensorflow"
ln -s libtensorflow_cc.so.2        libtensorflow_cc.so
ln -s libtensorflow_framework.so.2 libtensorflow_framework.so
cd <workspace>
```

### Step 6 -- Set Environment Variables

```bash
cd <workspace>/ZENTF_v5.2.1_C++_SOURCE_BUILD

# Sets TF_CC_API_ZENDNN_ROOT, LIBRARY_PATH, LD_LIBRARY_PATH,
# and creates the libomp.so.5 symlink.
source zentf_cc_api_setup.sh

# Enables ZenDNN plug-in and sets recommended performance tuning variables
# (KMP, OMP, ZENDNNL_MATMUL_ALGO, etc.).
source zentf_env_setup.sh
```

> **Note:** The `zentf_env_setup.sh` ships with OMP defaults tuned for AMD Turin.
> Adjust `OMP_NUM_THREADS` and related variables to match your hardware.

### Step 7 -- Compile and Run the Sample

Please follow steps in [Build C++ inference application](../../examples/c++/README.md) to build and run the sample inference application.

---

## Troubleshooting

| Issue | Resolution |
|---|---|
| `./configure` fails to find TensorFlow | Ensure TensorFlow is installed in the active conda env, or set `TF_HEADER_DIR` / `TF_SHARED_LIBRARY_DIR` manually. |
| Bazel version mismatch | Check `.bazelversion` in the repo and install the required Bazel version. |
| `libtensorflow_cc.so: cannot open shared object file` | Run `source zentf_cc_api_setup.sh` and verify `LD_LIBRARY_PATH` includes the TensorFlow library directory. |
| Linker errors during `g++` compilation | Ensure the `-rpath` and `-L` flags point to the correct directory containing the TensorFlow `.so` files. |
| Poor inference performance | Review and tune `OMP_NUM_THREADS`, `KMP_AFFINITY`, and `ZENDNNL_MATMUL_ALGO` in `zentf_env_setup.sh` for your CPU. |
