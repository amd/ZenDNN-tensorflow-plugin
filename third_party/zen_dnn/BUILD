load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [
        "//tensorflow_plugin:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

config_setting(
    name = "build_with_zendnn",
    define_values = {
        "build_with_zendnn": "true",
        "build_with_openmp": "true",
    },
    visibility = ["//visibility:public"],
)

filegroup(
    name = "LICENSES",
    srcs = [
        "LICENSE",
    ],
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)
