load("//tensorflow_plugin:amd_cpu_plugin.bzl", "tf_copts")
load("//third_party/zen_dnn:build_defs.bzl", "zendnn_deps")

package(licenses = ["notice"])  # Apache 2.0

cc_library(
    name = "zen_conv2d_op",
    srcs = ["zen_conv2d_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_matmul_ops",
    srcs = ["zen_matmul_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_batch_matmul_ops",
    srcs = ["zen_batch_matmul_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_pooling_ops",
    srcs = ["zen_pooling_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_softmax_op",
    srcs = ["zen_softmax_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_fused_batchnorm_ops",
    srcs = ["zen_fused_batchnorm_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_transpose_op",
    srcs = ["zen_transpose_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_reshape_op",
    srcs = ["zen_reshape_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

ZEN_OPS = [
    ":zen_conv2d_op",
    ":zen_matmul_ops",
    ":zen_softmax_op",
    ":zen_pooling_ops",
    ":zen_fused_batchnorm_ops",
    ":zen_transpose_op",
    ":zen_batch_matmul_ops",
    ":zen_reshape_op",
]

cc_library(
    name = "shape_inference_fns",
    srcs = ["shape_inference_fns.cc"],
    hdrs = ["shape_inference_fns.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = ["@local_config_tf//:tf_header_lib"],
)

cc_library(
    name = "zen_ops_impl",
    srcs = ["zen_ops_init.cc"],
    hdrs = ["zen_ops_init.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_gcc//:framework_lib",
        "@local_config_tf//:tf_header_lib",
    ] + ZEN_OPS,
    alwayslink = True,
)
