workspace(name = "org_tensorflow_plugin")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//third_party:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("3.1.0")

load("//tensorflow_plugin:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

load("//tensorflow_plugin:workspace.bzl", "amd_cpu_plugin_workspace", "clean_dep")

amd_cpu_plugin_workspace()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()
load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")
rules_cc_dependencies()

# Load rules_foreign_cc dependencies with required CMake.
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies(
    cmake_version = "3.27.9",  # Specify CMake version.
)

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()
