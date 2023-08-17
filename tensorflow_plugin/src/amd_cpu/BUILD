package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "amd_cpu_opkernels",
    srcs = ["amd_cpu_opkernels.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/kernels/zendnn:zen_kernels_impl",
        "//tensorflow_plugin/src/amd_cpu/ops/zendnn:zen_ops_impl",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)
