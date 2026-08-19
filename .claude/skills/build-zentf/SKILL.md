---
name: build-zentf
description: Build and verify the TensorFlow-ZenDNN plug-in (zentf) from source for the Python, C++, or Java interface on AMD EPYC CPUs. ALWAYS ask the user for the target TensorFlow version FIRST — it drives the whole build (which tensorflow is pip-installed, which tf_2.NN config ./configure selects, and which pinned ZenDNN tarball bazel fetches). Validate the version against version_configs/tf_2.* read fresh from the repo (never hardcoded). Python builds a pip wheel; C++/Java build the libamdcpu_plugin_cc.so shared library. The ZenDNN backend is fetched by bazel at the pinned zendnnl_repo tag as-is (no local submodule/symlink).
version: 0.1.0
---

# Build zentf (TensorFlow-ZenDNN plug-in)

zentf is the ZenDNN plug-in for TensorFlow. It has three user interfaces —
**Python**, **C++**, **Java** — that share a common front-end (confirm TF
version, conda env, `pip install tensorflow`, `./configure`) then branch. Python
produces a **pip wheel** (`//tensorflow_plugin/tools/pip_package:build_pip_package`);
C++ and Java both use the **plug-in shared library**
(`//tensorflow_plugin:libamdcpu_plugin_cc.so`).

The ZenDNN backend is fetched by bazel from a **pinned tarball tag**
(`zendnnl_repo` in `version_configs/tf_<ver>/workspace.bzl`) — used as-is. There
is no local ZenDNN submodule/symlink to manage (unlike the PyTorch plugin).

## Sections
- [Flow](#flow)
- [Inputs to confirm (ask the user)](#inputs-to-confirm-ask-the-user)
- [ALWAYS: ask + validate the TensorFlow version FIRST](#always-ask--validate-the-tensorflow-version-first)
- [Stage 1 - Environment prep](#stage-1---environment-prep)
- [Stage 2 - Configure](#stage-2---configure)
- [Interface branch: Python](#interface-branch-python)
- [Interface branch: C++](#interface-branch-c)
- [Interface branch: Java](#interface-branch-java)
- [Stage 5 - Enable + verify](#stage-5---enable--verify)
- [Stage 6 - Benchmark (optional)](#stage-6---benchmark-optional)
- [Gotchas](#gotchas)

## Flow

Shared front-end (TF version → env → `./configure`), then a three-way branch on
the interface, then a shared enable-and-verify tail.

The full flowchart lives in **[`build-zentf-flow.mmd`](build-zentf-flow.mmd)**
next to this file. Every gate in it has an explicit STOP terminal — there are no
retry loops.

In brief: confirm the TF version first and validate it against
`version_configs/tf_2.*` → Stage 1 env prep (conda env, `pip install
tensorflow==<ver>`, bazel, gcc) → Stage 2 `./configure` (which selects the
`tf_2.NN` config and pins the `zendnnl_repo` tarball) → branch on the interface
(Python wheel / C++ `libamdcpu_plugin_cc.so` / Java on top of that `.so`) →
Stage 5 enable and verify the `ZenDNN custom operations are on` log line →
Stage 6 optional clean benchmark.

## Inputs to confirm (ask the user)
1. **TensorFlow version** — the **primary, mandatory prompt** (see next
   section). Ask this FIRST, before any other step. Everything downstream
   depends on it.
2. **Interface(s)**: `Python`, `C++`, and/or `Java`. Build steps branch here.
3. **OS family**: Ubuntu vs RHEL/Fedora/AlmaLinux/CentOS. The RHEL family needs
   `export ZENDNNL_MANYLINUX_BUILD=1` before building from source.
4. **git ref**: the repo branch/tag to build (default: current checkout /
   `main`). The repo defaults to `main`; released versions live on `v<ver>`
   branches (e.g. `git checkout v2.21.0.0`).

## ALWAYS: ask + validate the TensorFlow version FIRST
**Do not run any build step until the TF version is confirmed AND validated.**
The version drives four things: the `pip install tensorflow==<ver>`, the
`tf_2.NN` config `./configure` selects, the pinned ZenDNN tarball bazel fetches,
and the wheel filename.

Read the supported set **fresh from the repo — never hardcode it** (it drifts
with each zentf release). Enumerate the available per-version configs:
```bash
ls -d version_configs/tf_2.* 2>/dev/null      # e.g. tf_2.19  tf_2.20  tf_2.21
grep -n 'build compatible with TensorFlow' README.md   # states the current range
```
- The README currently states zentf is **build compatible with TensorFlow
  2.16.0 through 2.21.0** (example only — grep fresh; it changes per release).
- `./configure` auto-detects the installed TF and maps `2.NN` -> the matching
  `version_configs/tf_2.NN/` config bundle.

**Auto-suggest a default:** if the user is unsure, offer the **highest**
`version_configs/tf_2.*` available (the newest supported config) as the default.

**STOP conditions:**
- The chosen `2.NN` has **no** matching `version_configs/tf_2.NN/` dir → stop and
  warn: unsupported TF version for this checkout (reconfirm or change ref).
- The full `MAJOR.MINOR.PATCH` is outside the README's stated compatible range →
  stop and warn before installing.

## Stage 1 - Environment prep
Uses the **confirmed** TF version throughout.
```bash
# Python 3.10-3.13 supported; 3.10 shown as an example.
conda create -n tf-v<ver>-zentf-env python=3.10 -y
conda activate tf-v<ver>-zentf-env
pip install tensorflow==<confirmed-ver>
# verify the installed version equals the one requested:
python3 -c "import tensorflow as tf; print(tf.__version__)"   # MUST == <confirmed-ver>
```
Toolchain requirements (from the repo prereqs / C++ build guide):
- **Bazel 7.7.0** (`./configure` sets `.bazelversion` to this; install it if
  missing).
- **gcc >= 13.1** (per the ZenDNN user guide), **git >= 1.8**.
- **RHEL family only:** `export ZENDNNL_MANYLINUX_BUILD=1`.
- **Java interface additionally needs:** Maven 3.6+, JDK 11+, `JAVA_HOME` set to
  the JDK path (e.g. `/usr/lib/jvm/java-11-openjdk-amd64` on Ubuntu), and
  GLIBC >= 2.33.

**STOP** if `tf.__version__` does not match the requested version, or if Bazel /
gcc / (for Java) Maven/JDK are missing.

## Stage 2 - Configure
```bash
cd <ZenDNN_TensorFlow_Plugin>
git checkout <ref>                    # optional: v<ver> branch or main
bazel clean --expunge --async
./configure
```
`./configure` auto-detects the python env and installed TF, copies the matching
`tf_2.NN` bundle into place, and sets `.bazelversion` to 7.7.0. It prints:
```
Configuring build for TensorFlow 2.NN (config: tf_2.NN)
...
Version configuration complete for TensorFlow 2.NN
```
**Verify** the `2.NN` in that line matches the confirmed TF version. Answer the
prompts (MPI: default `N`; optimization flags: default `-march=native
-Wno-sign-compare`).

**STOP** if the config version does not match, or if configure cannot find TF →
set `TF_HEADER_DIR` / `TF_SHARED_LIBRARY_DIR` manually and re-run.

---

## Interface branch: Python
Produces and installs the zentf pip wheel.
```bash
bazel build -c opt //tensorflow_plugin/tools/pip_package:build_pip_package \
    --verbose_failures --copt=-Wall --copt=-Werror --spawn_strategy=standalone
# generate the wheel into the repo root:
bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package .
# install by glob — the filename is version/python-tag pinned, do not hardcode:
pip install zentf-*.whl
```
`scripts/zentf_setup.sh` automates configure + build + wheel + install + env +
sample (it redirects bazel output to `bazel_build_output.txt`). You may run it
end-to-end instead of the manual steps once the TF version is confirmed:
```bash
source scripts/zentf_setup.sh
```

**Verify:**
```bash
python -c "import zentf; print(zentf.__version__)"
python -c "import zentf; print(*zentf.__config__.split(chr(10)), sep=chr(10))"
```
**STOP** on build failure → inspect `bazel_build_output.txt` (or the console with
`--verbose_failures`). Then go to [Stage 5 - Enable + verify](#stage-5---enable--verify).

## Interface branch: C++
Builds the plug-in shared library and assembles a self-contained C++ package.
Follows `scripts/c++/BUILD_FROM_SOURCE.md`.
```bash
# 1. build the plug-in .so
bazel build -c opt //tensorflow_plugin:libamdcpu_plugin_cc.so \
    --verbose_failures --copt=-Wall --copt=-Werror --spawn_strategy=standalone
# → bazel-bin/tensorflow_plugin/libamdcpu_plugin_cc.so

# 2. assemble the package (name mirrors the release convention)
cd <workspace>
PKG=ZENTF_v<ver>_C++_SOURCE_BUILD
mkdir -p $PKG/lib-tensorflow-plugins $PKG/examples
cp ZenDNN_TensorFlow_Plugin/bazel-bin/tensorflow_plugin/libamdcpu_plugin_cc.so \
   $PKG/lib-tensorflow-plugins/
cp ZenDNN_TensorFlow_Plugin/bazel-bin/tensorflow_plugin/libamdcpu_plugin_cc.so.runfiles/llvm_openmp/libiomp5.so \
   $PKG/lib-tensorflow-plugins/
cp ZenDNN_TensorFlow_Plugin/scripts/c++/zentf_cc_api_setup.sh $PKG/
cp ZenDNN_TensorFlow_Plugin/scripts/zentf_env_setup.sh        $PKG/
cp ZenDNN_TensorFlow_Plugin/examples/c++/sample_inference.cpp  $PKG/examples/

# 3. extract TF headers + shared libs from the matching wheel
TF_VERSION=<confirmed-ver>
mkdir -p tf_wheel_tmp
pip download "tensorflow==${TF_VERSION}" --no-deps --only-binary=:all: -d tf_wheel_tmp
unzip tf_wheel_tmp/tensorflow*.whl -d "$PKG/tensorflow_${TF_VERSION}"
rm -rf tf_wheel_tmp

# 4. create the required .so symlinks
cd "$PKG/tensorflow_${TF_VERSION}/tensorflow"
ln -sf libtensorflow_cc.so.2        libtensorflow_cc.so
ln -sf libtensorflow_framework.so.2 libtensorflow_framework.so
cd <workspace>

# 5. set library paths + env (also creates the libomp.so.5 symlink)
cd "$PKG"
source zentf_cc_api_setup.sh      # sets TF_CC_API_ZENDNN_ROOT, LIBRARY_PATH, LD_LIBRARY_PATH
source zentf_env_setup.sh         # enables zentf + perf tuning vars
```
**Verify** by compiling and running the sample (per `examples/c++/README.md`):
```bash
g++ examples/sample_inference.cpp -o sample_inference \
    -I./tensorflow_${TF_VERSION}/tensorflow/include \
    -L./tensorflow_${TF_VERSION}/tensorflow/ \
    -ltensorflow_framework -ltensorflow_cc \
    -Wl,-rpath=./tensorflow_${TF_VERSION}/tensorflow/ -std=c++17
# e.g. resnet50: ./sample_inference <resnet50_v1.pb> input_tensor softmax_tensor 1280 224 224 3
```
**STOP** on `libtensorflow_cc.so: cannot open shared object file` or linker
errors → verify `LD_LIBRARY_PATH`/`-rpath`/`-L` point at the extracted TF `.so`
dir and that `source zentf_cc_api_setup.sh` ran. Then go to
[Stage 5 - Enable + verify](#stage-5---enable--verify).

## Interface branch: Java
Java reuses the C++ plug-in `.so`, loaded at runtime via TensorFlow-Java, built
from source. Follows `scripts/java/README.md`. **The TensorFlow-Java version/ref
MUST match the selected TF version's C++ ABI:** `libamdcpu_plugin_cc.so` is
compiled against a specific TF version's C++ ABI, so a mismatched TensorFlow-Java
build will fail to load or link the plug-in. `scripts/java/README.md` currently
documents TensorFlow-Java **1.2.0-SNAPSHOT** (for TF 2.21.0) — treat that as the
example for TF 2.21.0, not a fixed value for every TF version.
```bash
# prereqs: Bazel 7.7.0, Maven 3.6+, JDK 11+, JAVA_HOME set, GLIBC >= 2.33
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64   # adjust to your JDK

# 1. build TensorFlow-Java (clones tensorflow/java, checks out the tensorflow/java
#    ref matching the SELECTED TF version's C++ ABI, runs `mvn clean install`).
#    MUST be sourced (script uses `return`).
#    NOTE: scripts/java/build_tf_java.sh pins a `git checkout <commit>` that
#    corresponds to TF 2.21.0 (TensorFlow-Java 1.2.0-SNAPSHOT). Do NOT reuse that
#    commit for a different TF version — it is tied to TF 2.21.0's C++ ABI.
#    For another selected TF version, check out the tensorflow/java ref whose
#    bundled TF C++ ABI matches your version (see the tensorflow/java release
#    tags / TF-version mapping) before `mvn clean install`.
cd ZenDNN_TensorFlow_Plugin/scripts/java
source build_tf_java.sh

# 2. build the C++ plug-in .so (see the C++ branch, step 1) if not already built

# 3. point the loader at the plug-in libraries
export LD_LIBRARY_PATH=<path to zentf C++ pkg>/lib-tensorflow-plugins:$LD_LIBRARY_PATH
source <path to zentf C++ pkg>/zentf_env_setup.sh
```
Load the plug-in in application code:
```java
import static org.tensorflow.internal.c_api.global.tensorflow.TF_LoadPluggableDeviceLibrary;
import org.tensorflow.internal.c_api.TF_Library;
import org.tensorflow.internal.c_api.TF_Status;

private static void load_zentf(String filename) {
    TF_Status status = TF_Status.newStatus();
    TF_Library h = TF_LoadPluggableDeviceLibrary(filename, status); // "libamdcpu_plugin_cc.so" on LD_LIBRARY_PATH
    status.throwExceptionIfNotOK();
}
```
**Verify** with the Wide&Deep example (per `examples/java/README.md`):
```bash
cd ZenDNN_TensorFlow_Plugin/examples/java
mvn clean package
java -cp target/tensorflow-benchmark-0.1-jar-with-dependencies.jar \
     org.tensorflow.benchmark.RunWideDeeplarge <wide_deep .pb> <batch_size>
```
**STOP** on Maven/bazel/JDK failure → fix `JAVA_HOME`, Maven, Bazel 7.7.0, or
GLIBC; if `tf_java.git` already exists the script prompts to remove it. Then go
to [Stage 5 - Enable + verify](#stage-5---enable--verify).

---

## Stage 5 - Enable + verify
Enable the plug-in and run the interface's sample. **This is the success
criterion for the build.**
```bash
export TF_ENABLE_ZENDNN_OPTS=1      # enable ZenDNN optimizations
export TF_ENABLE_ONEDNN_OPTS=0      # disable oneDNN so zentf drives the ops
source scripts/zentf_env_setup.sh   # sets KMP/OMP + ZENDNNL_MATMUL_ALGO defaults
```
Run the sample for the interface built:
- **Python:** `python tests/softmax.py`
- **C++:** `./sample_inference <model.pb> ...` (resnet50 / mobilenetv1)
- **Java:** the Wide&Deep run above

**Pass criteria:**
1. The log contains `ZenDNN custom operations are on` (confirms zentf is active,
   not stock TF). It comes from `port.cc` at startup.
2. The sample prints its expected output (softmax vector / FPS line / batch
   output) and exits 0.

**STOP** if the `ZenDNN custom operations are on` line is absent → zentf is not
active: confirm `TF_ENABLE_ZENDNN_OPTS=1`, `TF_ENABLE_ONEDNN_OPTS=0`, that the
wheel/`.so` for the active interface is installed/loaded, and rebuild if needed.

## Stage 6 - Benchmark (optional)
Only if the user wants performance numbers. Run a **clean** run (no debug/verbose
logging). Tune per hardware — the `zentf_env_setup.sh` defaults target AMD Turin:
- `OMP_NUM_THREADS` — set to the physical core count (128 Turin default in the
  script; adjust for Genoa etc.).
- `KMP_AFFINITY=granularity=fine,compact,1,0`, `OMP_PROC_BIND`, `OMP_WAIT_POLICY`
  — already exported by the setup script.
- `ZENDNNL_MATMUL_ALGO`: **1** for FP32 / direct BF16 models; **4 or 5** for AMP
  models (when `TF_ZENDNN_PLUGIN_BF16=1`). The setup script auto-selects 4 when
  `TF_ZENDNN_PLUGIN_BF16=1`, else 1.
- Python-level threading for the Python interface: `TF_NUM_INTRAOP_THREADS`,
  `TF_NUM_INTEROP_THREADS`.

Report throughput (FPS) / latency from the sample output (e.g. the C++ sample
prints `FPS for N images: ...`).

## Gotchas
- **Ask the user for the TF version FIRST and let it drive everything** — never
  hardcode it. Validate against `version_configs/tf_2.*` (fresh) and confirm
  `./configure` printed the matching `tf_2.NN` config.
- Supported build range is stated in `README.md` (currently TF 2.16.0-2.21.0) —
  grep it fresh; it changes per release.
- **RHEL/Fedora/Alma/CentOS:** `export ZENDNNL_MANYLINUX_BUILD=1` before building.
- **Wheel filename is version/python-tag pinned** — install by glob
  (`pip install zentf-*.whl`), don't hardcode `cp310` etc.
- **C++** needs the TF headers/libs extracted from the pip wheel, the two
  `libtensorflow_{cc,framework}.so` symlinks, and the `libomp.so.5` symlink
  (created by `zentf_cc_api_setup.sh`) — missing any → runtime `.so` load errors.
- **Java** `build_tf_java.sh` must be **sourced** (it uses `return`); it needs
  GLIBC >= 2.33 + `JAVA_HOME`. It checks out a tensorflow/java commit **pinned to
  TF 2.21.0** — that ref must match the selected TF version's C++ ABI, so change
  it (do not reuse the 2.21.0 commit) when building for a different TF version.
- **C++ and Java share the same `libamdcpu_plugin_cc.so`** — build it once.
- **Always `bazel clean --expunge`** before a rebuild (a stale cache can pin an
  old config/compiler).
- Enable requires **both** `TF_ENABLE_ZENDNN_OPTS=1` **and**
  `TF_ENABLE_ONEDNN_OPTS=0`; the presence of `ZenDNN custom operations are on` in
  the log is the activation proof.
- The ZenDNN backend is a **pinned bazel tarball** (`zendnnl_repo` in
  `version_configs/tf_<ver>/workspace.bzl`) — used as-is; no submodule to init.
