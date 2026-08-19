# AGENTS.md

Guidance for AI coding agents working in the **ZenDNN TensorFlow Plug-in
(zentf)** repository. This file captures always-useful context about the repo
and its architecture, and points to specialized agent skills for detailed,
step-by-step workflows. Read this first; follow a skill when your task matches
one below.

## What this repo is

**zentf** is the **ZenDNN plug-in for TensorFlow**. It integrates the ZenDNN
inference library into TensorFlow so that TensorFlow inference on AMD EPYC™ CPUs
dispatches supported operations to ZenDNN-optimized kernels. It is built as a
TensorFlow **PluggableDevice / graph-optimizer plug-in**, not a TensorFlow fork:
stock TensorFlow loads the plug-in at runtime and hands eligible ops to it.

- The plug-in registers a custom graph optimizer and custom op kernels for the
  CPU device. When enabled, startup logs print `ZenDNN custom operations are on`
  (emitted from `port.cc`) — this line is the activation proof.
- zentf exposes **three user interfaces** that share a common front-end
  (confirm TF version, conda env, `pip install tensorflow`, `./configure`) then
  branch:
  - **Python** — builds and installs a pip **wheel** (`zentf-*.whl`).
  - **C++** — builds the plug-in shared library
    **`libamdcpu_plugin_cc.so`** and assembles a self-contained C++ package.
  - **Java** — reuses the same `libamdcpu_plugin_cc.so`, loaded at runtime via
    TensorFlow-Java.
- The **ZenDNN backend** is not vendored as a submodule. It is fetched by bazel
  from a **pinned tarball** (the `zendnnl_repo` archive declared in
  `version_configs/tf_<ver>/workspace.bzl`) and used as-is.

zentf follows TensorFlow's version numbering: the version format is
`TF_MAJOR.TF_MINOR.TF_PATCH.PLUGIN_PATCH` (e.g. `2.21.0.1` targets TensorFlow
`2.21.0`). For the public overview, supported versions, and install
instructions, see `README.md`.

## Repository layout

```
ZenDNN_TensorFlow_Plugin
|- configure / configure.py : build configuration entry point. Auto-detects the
|                             installed TF version and copies the matching
|                             version_configs/tf_2.NN bundle into place.
|- version_configs/         : per-TF-version build configs (tf_2.19, tf_2.20,
|                             tf_2.21, ...). Each holds WORKSPACE, workspace.bzl,
|                             build_config_util.bzl, BUILD.tpl — including the
|                             pinned `zendnnl_repo` tarball for that TF version.
|- WORKSPACE                : bazel workspace (populated by ./configure from the
|                             selected version_configs bundle).
|- tensorflow_plugin/       : the plug-in source.
|   |- src/amd_cpu/         : native plug-in implementation.
|   |   |- graph/           : graph optimizer (cpu_optimizer, remapper,
|   |   |                     auto_mixed_precision, graph_view, zendnn passes).
|   |   |- kernels/zendnn/  : ZenDNN-backed op kernels.
|   |   |- ops/             : op registrations.
|   |   |- util/            : plug-in utilities and build_config.
|   |- python/             : Python package (zentf) sources.
|   |- tools/pip_package/  : wheel build rule (build_pip_package).
|- examples/               : per-interface inference examples (c++, java, python).
|- scripts/                : build/setup scripts.
|   |- zentf_setup.sh       : end-to-end Python configure+build+wheel+install+env.
|   |- zentf_env_setup.sh   : enable zentf + KMP/OMP + ZENDNNL perf-tuning vars.
|   |- c++/                 : C++ build guide (BUILD_FROM_SOURCE.md) + setup.
|   |- java/                : TensorFlow-Java build script + guide.
|- tests/                  : per-op TensorFlow test scripts (softmax, matmul, ...).
|- third_party/            : bazel repo rules and dependency BUILD files.
|- .claude/skills/         : agent skills (see below).
```

## Build system

- **Bazel 7.7.0** drives the build. `./configure` sets `.bazelversion` to
  `7.7.0`, auto-detects the active Python env and installed TensorFlow, maps the
  detected `2.NN` to the matching `version_configs/tf_2.NN/` bundle, and copies
  that bundle's `WORKSPACE`, `workspace.bzl`, `build_config_util.bzl`, and
  `BUILD.tpl` into place. It prints
  `Configuring build for TensorFlow 2.NN (config: tf_2.NN)` and
  `Version configuration complete for TensorFlow 2.NN`.
- The **TensorFlow version is the primary input** and drives everything: which
  TensorFlow is `pip install`ed, which `tf_2.NN` config `./configure` selects,
  which pinned `zendnnl_repo` tarball bazel fetches, and the wheel filename.
  README states the current build-compatible TensorFlow range (grep it fresh —
  it changes per release).
- **Python** target:
  `//tensorflow_plugin/tools/pip_package:build_pip_package` → then
  `bazel-bin/.../build_pip_package .` writes the `zentf-*.whl` to the repo root.
- **C++ / Java** target: `//tensorflow_plugin:libamdcpu_plugin_cc.so` →
  `bazel-bin/tensorflow_plugin/libamdcpu_plugin_cc.so`. C++ and Java share the
  same `.so` — build it once.
- The **ZenDNN backend** (`zendnnl_repo`) is fetched by bazel from the pinned
  tarball in the selected `version_configs/tf_<ver>/workspace.bzl` — there is no
  local ZenDNN submodule/symlink to initialize.
- RHEL/Fedora/AlmaLinux/CentOS: `export ZENDNNL_MANYLINUX_BUILD=1` before
  building from source.

For the detailed and authoritative build workflow, **use the `build-zentf`
skill** rather than reconstructing commands by hand. The C++ build guide lives
at `scripts/c++/BUILD_FROM_SOURCE.md`; the Java guide at `scripts/java/README.md`.

## Enabling the plug-in

At runtime, both of these must be set for zentf to drive the ops:

```
export TF_ENABLE_ZENDNN_OPTS=1
export TF_ENABLE_ONEDNN_OPTS=0
```

`scripts/zentf_env_setup.sh` sets these plus KMP/OMP threading and
`ZENDNNL_MATMUL_ALGO` performance defaults. The presence of
`ZenDNN custom operations are on` in the log confirms activation.

## Agent skills

Specialized, step-by-step workflows live under `.claude/skills/`. When a task
matches one of these, **follow the skill** — it is the authoritative,
stop-on-failure procedure. Do not duplicate or paraphrase its commands here.

- **`build-zentf`** — `.claude/skills/build-zentf/SKILL.md`
  Build and verify zentf from source for the **Python**, **C++**, or **Java**
  interface on AMD EPYC CPUs. It ALWAYS asks the user for the target TensorFlow
  version FIRST (the mandatory primary prompt) and validates it against
  `version_configs/tf_2.*` read fresh from the repo; that version then drives the
  `pip install tensorflow`, the `tf_2.NN` config `./configure` selects, and the
  pinned `zendnnl_repo` tarball bazel fetches. Python builds/installs the pip
  wheel; C++/Java build `libamdcpu_plugin_cc.so`. It ends by enabling the plug-in
  and running the interface's sample, treating `ZenDNN custom operations are on`
  plus correct sample output as the success criterion.

### Working principle for skills

The build skill enforces a strict **stop-on-failure policy**: if any command
fails, stop and report the failing step, the exact command, and the first real
error line — do not auto-retry, auto-fix, or continue on your own. Honor that
policy when executing this workflow.

## Conventions

- Keep documentation factual: do not invent APIs, flags, or version numbers.
- Prefer editing existing files over creating new ones.
- Commit messages describe only the change — no AI/tool attribution. This repo
  has a commit-msg hook that requires a `Signed-off-by` line, so commit with
  `git commit -s`.
