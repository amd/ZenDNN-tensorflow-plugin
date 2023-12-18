load(
    "//tensorflow_plugin/src/amd_cpu/util:build_config.bzl",
    "tf_proto_library",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "amd_cpu_opkernels",
    srcs = ["amd_cpu_opkernels.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/graph:cpu_optimizer",
        "//tensorflow_plugin/src/amd_cpu/kernels/zendnn:zen_kernels_impl",
        "//tensorflow_plugin/src/amd_cpu/ops/zendnn:zen_ops_impl",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

tf_proto_library(
    name = "protos_all",
    srcs = [],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow_plugin/src/amd_cpu/util/protobuf:for_core_protos",
    ],
    visibility = ["//visibility:public"],
)
